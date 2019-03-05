"""Train a policy network with supervision from a planner and hard negative
mining."""

import rpyc
from enum import Enum
from functools import lru_cache
from itertools import cycle, islice
from random import shuffle
import re
import threading
from typing import (  # noqa
    Any, Generator, List, Set, Tuple, Dict, Iterable, Union, TYPE_CHECKING)

from rllab.envs.base import Env, Step
import tensorflow as tf
import numpy as np
import tqdm

from ssipp_interface import Planner, format_state_for_ssipp, set_up_ssipp
from rllab_interface import FPGObservation, unwrap_env, create_environment
from mdpsim_utils import parse_problem_args
from models import (BoundAction, ProblemMeta, get_problem_meta,
                    get_domain_meta, PropNetworkWeights)
from multiprob import to_local
from prof_utils import can_profile

if TYPE_CHECKING:
    # this is never executed at runtime, it's just for MyPy
    import mdpsim  # noqa
    import fpg  # noqa

# Q-values for all actions
QVs = Tuple[Tuple[BoundAction, float], ...]
# (state, action) pair
StateAction = Tuple[FPGObservation, BoundAction]
# (state, Q-values)
StateQVs = Tuple[FPGObservation, QVs]
# Bunch of states seen and actions taken
Path = List[StateAction]


def cross_entropy(act_dist, labels):
    # This operates on probabilities, not logits, using approach from
    # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L2725-L2753
    # Logits would probably be more stable but are harder to deal with in my
    # masked softmax formulation.
    # TODO: do I even need this given I'm using softmax? Should be normalised
    # already, no? Maybe replace with assertion.
    sums = tf.reduce_sum(act_dist, axis=-1)
    all_good = tf.reduce_all(tf.abs(sums - 1) < 1e-2, axis=None)
    control_data = [sums, tf.reduce_min(sums), tf.reduce_max(sums)]
    check_normed_op = tf.Assert(all_good, control_data, name='check_normed')
    # act_dist /= tf.reduce_sum(act_dist, axis=-1, keep_dims=True)
    with tf.control_dependencies([check_normed_op]):
        eps = 1e-8
        clipped = tf.clip_by_value(act_dist, eps, 1 - eps)
        return tf.reduce_sum(-labels * tf.log(clipped), axis=-1)


@can_profile
def collect_paths(rllab_env: Env,
                  prob_meta: ProblemMeta,
                  get_action: Any,
                  n_paths: int,
                  max_len: int) -> Tuple[List[Path], float]:
    """Get a few (state, action) paths from initial state to the goal or to a
    dead end. Also gives list of states visited (w/ MDPSim)."""
    assert max_len > 0

    paths = []
    uenv = unwrap_env(rllab_env)
    ospace = uenv.observation_space
    tot_rew = 0.0

    for path_num in range(n_paths):
        path = []
        obs = rllab_env.reset()
        done = False

        for step_num in range(max_len):
            # TODO: some fraction of the time I should take a random action,
            # and some fraction of the time I should take an LRTDP-recommended
            # action. Maybe---need to think about the optimal thing to do here.
            action, _ = get_action(obs)
            path.append((ospace.unflatten(obs),
                         prob_meta.bound_acts_ordered[action]))
            # done is true in both goal states and recognised dead ends
            # info contains 'goal_reached' and 'progress', which can both be
            # useful
            obs, rew, done, info = rllab_env.step(action)
            tot_rew += rew
            if done:
                break
        else:
            # Need to do this outside loop for last step so that we can get
            # both beginning and end. Maybe it's better just to ignore last
            # step?
            path.append((ospace.unflatten(obs),
                         prob_meta.bound_acts_ordered[action]))
        paths.append(path)

    npaths = len(paths)
    succ_rate = tot_rew / npaths

    return paths, succ_rate


@can_profile
def planner_trace(planner: Planner,
                  prob_meta: ProblemMeta,
                  mdpsim_problem: 'mdpsim.Problem',
                  obs: FPGObservation) -> List[StateQVs]:
    """Extract (s, [q*]) pairs for all s reachable from (state) under some
    (arbitrary) optimal policy."""
    pairs = []  # type: List[StateQVs]
    props_true = obs.props_true
    start_state_string = format_state_for_ssipp(props_true)
    ssipp_problem = planner.problem
    start_ssipp_state \
        = ssipp_problem.get_intermediate_state(start_state_string)
    # not sure how expensive this is, but IIRC not very
    pol_dict, overflow = planner.extract_policy(start_ssipp_state)
    if overflow:
        print("WARNING: extract_policy exceeded max. returned state count!")
    prop_re = re.compile(r'(\([^ \)]+(?: [^ \)]+)*\))')
    for ssipp_state, _ in pol_dict.items():
        state_str = ssipp_problem.string_repr(ssipp_state)
        true_bound_props = []
        # converting props to BoundProps and back like this is probably not
        # necessary now, but I'm keeping it anyway because it serves as nice
        # Python-side error checking.
        for prop_s in prop_re.findall(state_str):
            bound_prop = prob_meta.bound_prop_by_name(prop_s)
            true_bound_props.append(bound_prop)
        mdpsim_state_str = ', '.join(' '.join((p.pred_name, ) + p.arguments)
                                     for p in true_bound_props)
        mdpsim_state = mdpsim_problem.intermediate_atom_state(mdpsim_state_str)
        new_props_true = mdpsim_problem.prop_truth_mask(mdpsim_state)
        new_acts_enabled = mdpsim_problem.act_applicable_mask(mdpsim_state)
        new_obs = FPGObservation(
            props_true=new_props_true, enabled_actions=new_acts_enabled)

        # get q-values for enabled actions
        act_strs = [ba.unique_ident for ba in prob_meta.bound_acts_ordered]
        q_values = planner.q_values(mdpsim_state_str, act_strs)
        assert len(act_strs) == len(q_values)
        qv_tuple = tuple(zip(prob_meta.bound_acts_ordered, q_values))

        pairs.append((new_obs, qv_tuple))
    return pairs


class ProblemServiceConfig(object):
    def __init__(self,
                 pddl_files,
                 init_problem_name,
                 heuristic_name=None,
                 use_lm_cuts=False,
                 dump_table_path=None,
                 dump_table_interval=None,
                 max_len=50,
                 teacher_heur='lm-cut'):
        # Things this needs to do, and parameters I need to pass to do those
        # things:
        # - Initialise mdpsim and ssipp (requires pddl_files, problem_name)
        # - Initialise data generators. This might be easiest to achieve with
        #   just a list of generator class names and arguments (although I
        #   still need to make sure those are actually deep copied, grumble
        #   grumble)
        self.pddl_files = pddl_files
        self.init_problem_name = init_problem_name
        self.heuristic_name = heuristic_name
        self.use_lm_cuts = use_lm_cuts
        self.dump_table_path = dump_table_path
        self.dump_table_interval = dump_table_interval
        self.max_len = max_len
        self.teacher_heur = teacher_heur


class PlannerExtensions(object):
    """Wrapper to hold references to SSiPP and MDPSim modules, and references
    to the relevant loaded problems (like the old ModuleSandbox). Mostly
    keeping this because it makes it convenient to pass stuff around, as I
    often need SSiPP and MDPSim at the same time."""

    def __init__(self, pddl_files, init_problem_name):
        print('Parsing %d PDDL files' % len(pddl_files))

        import mdpsim  # noqa: F811
        import ssipp  # noqa: F811

        self.mdpsim = mdpsim
        self.mdpsim_problem = parse_problem_args(self.mdpsim, pddl_files,
                                                 init_problem_name)
        self.problem_name = self.mdpsim_problem.name

        self.ssipp = ssipp
        self.ssipp_problem = set_up_ssipp(self.ssipp, pddl_files,
                                          self.problem_name)

        # these debug prints are so frequently useful that I haven't bothered
        # removing them
        print('\nInformation for problem "%s":' % self.mdpsim_problem.name)
        print('Propositions: %d' % len(self.mdpsim_problem.propositions))
        print('Ground actions: %d' % len(self.mdpsim_problem.ground_actions))
        mdpsim_domain = self.mdpsim_problem.domain
        print('Mdpsim_Domain: %s' % mdpsim_domain.name)
        print('Lifted actions: %d' % len(mdpsim_domain.lifted_actions))
        print('Predicates: %d' % len(mdpsim_domain.predicates))
        print('')

    @property
    def ssipp_dead_end_value(self):
        return self.ssipp.get_dead_end_value()


class RemoteEnv(Env):
    """Wraps a problem_service to remotely execute actions and generate
    perceptions on the server. Compatible with ordinary RLLab envs."""

    def __init__(self, problem_server):
        self._first_step = True
        self._problem_server = problem_server
        self._problem_service = problem_server.service
        spec = to_local(self._problem_service.get_env_spec())
        self._spec = spec

    @property
    def action_space(self):
        return self._spec.action_space

    @property
    def observation_space(self):
        return self._spec.observation_space

    def reset(self):
        remote_obs = self._problem_service.env_reset()
        return to_local(remote_obs)

    def step(self, action):
        if self._first_step:
            # first step, reset
            self.reset()
            self._first_step = False

        # rllab.envs.base.Step is a function which creates a namedtuple *also*
        # called rllab.envs.base.Step. That confuses the pickler, since it
        # expects to be able to find identical classes on the client side to
        # expand pickle back out into. Need to convert attribute-by-attribute
        # instead.
        remote_step = self._problem_service.env_step(action)
        observation, reward, done, info = map(to_local, remote_step)

        return Step(observation, reward, done, **info)

    def action_name(self, action_num):
        return self._problem_service.action_name(action_num)


def make_problem_service(config: ProblemServiceConfig):
    """Construct Service class for a particular problem. Note that we must
    construct classes, not instances (unfortunately), as there is no way of
    passing arguments to the service's initialisation code (AFAICT)."""

    class ProblemService(rpyc.Service):
        """Spools up a new Python interpreter and uses it to sandbox SSiPP and
        MDPSim. Can interact with this to train a Q-network."""

        # TODO: figure out some way of ensuring that arguments and return
        # values are deep-copied, or the whole impl will end up dog slow.

        def exposed_extend_replay(self, get_action, n_paths):
            """Extend the replay buffer using the given policy (represented as a
            function from flattened observation vectors to action numbenrs)."""
            n_paths = to_local(n_paths)
            return self.internal_extend_replay(get_action, n_paths)

        def exposed_batch_iter(self, batch_size, n_batches):
            """Sample <batch_size> elements from internal buffer."""
            batch_size = to_local(batch_size)
            n_batches = to_local(n_batches)
            # first convert replay buffer to a list so that we can shuffle and
            # take indices
            assert len(self.replay) > 0, 'need non-empty replay pool'
            ordered_buf = list(self.replay)
            shuffle(ordered_buf)  # in-place
            gen = cycle(ordered_buf)
            for batch_num in range(n_batches):
                rich_batch = list(islice(gen, batch_size))
                yield self.flatten_batch(rich_batch)

        def exposed_env_reset(self):
            return self.env_wrapped.reset()

        def exposed_action_name(self, action_num):
            action_num = to_local(action_num)
            return self.env_raw.action_name(action_num)

        def exposed_env_step(self, action):
            action = to_local(action)
            return self.env_wrapped.step(action)

        # note to self: RPyC doesn't support @property

        def exposed_get_ssipp_dead_end_value(self):
            return self.p.ssipp_dead_end_value

        def exposed_get_meta(self):
            """Get name, ProblemMeta and DomainMeta for the current problem."""
            return self.problem_meta, self.domain_meta

        def exposed_get_replay_size(self):
            return len(self.replay)

        def exposed_get_env_spec(self):
            # this will have to be deep-copied to actually work (I think)
            return self.env_wrapped.spec

        def exposed_get_obs_dim(self):
            return self.env_wrapped.observation_space.flat_dim

        def exposed_get_act_dim(self):
            return self.env_wrapped.action_space.flat_dim

        def exposed_get_obs_dtype_name(self):
            return self.env_wrapped.observation_space.dtype.name

        def exposed_get_dg_extra_dim(self):
            data_gens = self.env_raw.data_gens
            return sum([g.extra_dim for g in data_gens])

        def exposed_get_max_len(self):
            return self.max_len

        def exposed_get_problem_names(self):
            # fetch a list of all problems loaded by MDPSim
            return sorted(self.p.mdpsim.get_problems().keys())

        def exposed_get_current_problem_name(self):
            return self.p.problem_name

        def on_connect(self):
            self.p = PlannerExtensions(config.pddl_files,
                                       config.init_problem_name)
            mdpsim_p = self.p.mdpsim_problem
            self.domain_meta = get_domain_meta(mdpsim_p.domain)
            self.problem_meta = get_problem_meta(mdpsim_p, self.domain_meta)
            self.env_raw, self.env_wrapped = create_environment(
                problem_meta=self.problem_meta,
                planner_extensions=self.p,
                heuristic_name=config.heuristic_name,
                use_lm_cuts=config.use_lm_cuts,
                dump_table_path=config.dump_table_path,
                dump_table_interval=config.dump_table_interval)
            self.planner = Planner(self.p, 'lrtdp', config.teacher_heur)

            # maximum length of a trace to gather
            self.max_len = config.max_len
            # will hold (state, action) pairs to train on
            self.replay = set()  # type: Set[StateQVs]

        @lru_cache(None)
        def opt_pol_envelope(self, obs: FPGObservation) -> List[StateQVs]:
            """Get (s, a) pairs for optimal policy from given state."""
            return planner_trace(self.planner, self.problem_meta,
                                 self.p.mdpsim_problem, obs)

        def internal_extend_replay(self, get_action: Any, n_paths: int) \
                -> float:
            """Extend the supervision buffer with some new paths. Can probably make
            this more sophisticated by turning it into a least-recently-visited
            cache or something."""
            paths, succ_rate = collect_paths(
                rllab_env=self.env_wrapped,
                prob_meta=self.problem_meta,
                get_action=get_action,
                n_paths=n_paths,
                max_len=self.max_len)
            new_pairs = set()
            for path in paths:
                # ignore original action
                for obs, _ in path:
                    # I used to hard negative mine (only add to training set if
                    # net gets it wrong), but now I don't bother
                    new_pairs.update(self.opt_pol_envelope(obs))
            self.replay.update(new_pairs)
            return succ_rate

        def flatten_batch(self, rich_batch: List[StateQVs]) \
                -> Tuple[np.ndarray, np.ndarray]:
            rich_obs, rich_qvs = zip(*rich_batch)
            obs_tensor = self.env_raw.observation_space.flatten_n(rich_obs)
            qv_lists = []
            for qv_pairs in rich_qvs:
                qv_dict = dict(qv_pairs)
                qv_list = [
                    qv_dict[ba] for ba in self.problem_meta.bound_acts_ordered
                ]
                qv_lists.append(qv_list)
            qv_tensor = np.array(qv_lists)
            return obs_tensor, qv_tensor

    return ProblemService


class SupervisedObjective(Enum):
    ANY_GOOD_ACTION = 0
    MAX_ADVANTAGE = 1


class SupervisedTrainer:
    @can_profile
    def __init__(self,
                 problems: List['fpg.SingleProblem'],
                 weight_manager: PropNetworkWeights,
                 summary_writer: Any,
                 strategy: SupervisedObjective,
                 kl_coeff: Union[None, float],
                 batch_size: int=64,
                 lr: float=0.001) -> None:
        # this needs to be acquired before figuring out an action from NN
        self._get_act_lock = threading.RLock()
        # gets incremented to deal with TF
        self.batches_seen = 0
        self.problems = problems
        self.weight_manager = weight_manager
        self.summary_writer = summary_writer
        self.batch_size = batch_size
        self.batch_size_per_problem = max(batch_size // len(problems), 1)
        self.strategy = strategy
        self.kl_coeff = kl_coeff
        self.max_len = max(
            to_local(problem.problem_service.get_max_len())
            for problem in self.problems)
        self.tf_init_done = False
        self.lr = lr
        self._init_tf()

    # return sig structure: Generator[YieldType, SendType, ReturnType]
    # TODO: add deadline here
    @can_profile
    def train(self, max_epochs: int=100) \
            -> Generator[Tuple[float, float, int], bool, None]:
        """Train the network for a while."""
        assert self.tf_init_done, "Must call .init_tf() first!"

        tr = tqdm.trange(max_epochs, desc='epoch', leave=True)
        mean_loss = None

        for epoch_num in tr:
            # only extend replay by a bit each time
            succ_rates = self._extend_replays(max(25 // len(self.problems), 1))
            succ_rate = np.mean(succ_rates)
            replay_sizes = self._get_replay_sizes()
            replay_size = sum(replay_sizes)
            tr.set_postfix(
                succ_rate=succ_rate, net_loss=mean_loss, states=replay_size)
            self._log_op_value('succ-rate', succ_rate)
            self._log_op_value('replay-size', replay_size)
            # do a few batches of SGD (should keep us close to convergence)
            mean_loss = self._optimise(300)
            tr.set_postfix(
                succ_rate=succ_rate, net_loss=mean_loss, states=replay_size)
            keep_going = yield succ_rate, mean_loss, replay_size
            if not keep_going:
                print('.train() terminating early')
                break

    def _get_replay_sizes(self) -> List[int]:
        """Get the sizes of replay buffers for each problem."""
        rv = []
        for problem in self.problems:
            rv.append(to_local(problem.problem_service.get_replay_size()))
        return rv

    def _make_get_action(self, problem: 'fpg.SingleProblem'):
        """Make a function which takes a remote observation and yields an
        action number."""
        get_action = problem.policy.get_action

        # each get_action has an ephemeral cache (lasts only as long as the
        # closure does)
        cache = {}

        def inner(obs):
            obs = to_local(obs)
            try:
                # if this times out then something really screwy is going on
                self._get_act_lock.acquire(timeout=60 * 30)
                # each thread needs to have this call somewhere, per
                # https://www.tensorflow.org/versions/r0.12/api_docs/python/client/session_management
                with self.sess.as_default():
                    # make sure it's 1D (need different strategy for batch
                    # cache)
                    assert obs.ndim == 1
                    obs_bytes = obs.tostring()
                    if obs_bytes not in cache:
                        cache[obs_bytes] = get_action(obs)
                        return cache[obs_bytes]
                    return cache[obs_bytes]
            finally:
                self._get_act_lock.release()

        return inner

    @can_profile
    def _extend_replays(self, num_per_problem: int):
        """Extend the replays for //all// problems asynchronously."""
        # fire off extension methods
        results = []
        for problem in tqdm.tqdm(self.problems, desc='spawn extend'):
            get_action = self._make_get_action(problem)
            extend_replay = rpyc.async(problem.problem_service.extend_replay)
            result = extend_replay(get_action, num_per_problem)
            # apparently I need to keep hold of async ref according to RPyC
            # docs (it's weak or s.th). Also, I need a background thread to
            # serve each environment's requests (...this may break things
            # slightly).
            bg_thread = rpyc.utils.helpers.BgServingThread(
                problem.problem_server.conn)
            results.append((extend_replay, result, bg_thread))

        # Now we wait for results to come back. This is horribly inefficient
        # when some environments are much harder than others; oh well.
        succ_rates = []
        for _, result, bg_thread in tqdm.tqdm(results, desc='wait extend'):
            succ_rates.append(to_local(result.value))
            # always shut down cleanly
            bg_thread.stop()

        return succ_rates

    @can_profile
    def _instantiate_net(self, single_prob_instance: 'fpg.SingleProblem'):
        # create two placeholders
        problem_service = single_prob_instance.problem_service
        policy = single_prob_instance.policy
        obs_dim = to_local(problem_service.get_obs_dim())
        obs_dtype_name = to_local(problem_service.get_obs_dtype_name())
        ph_obs_var = tf.placeholder(
            shape=[None, obs_dim], name='observation', dtype=obs_dtype_name)
        act_dist = policy.dist_info_sym(
            ph_obs_var, summary_collections=['sl-activations'])['prob']
        act_dim = to_local(problem_service.get_act_dim())
        ph_q_values = tf.placeholder(
            shape=[None, act_dim], name='q_values', dtype='float32')

        loss_parts = []

        # now the loss ops
        if self.strategy == SupervisedObjective.ANY_GOOD_ACTION:
            best_qv = tf.reduce_min(ph_q_values, axis=-1, keep_dims=True)
            # TODO: is 0.01 threshold too big? Hmm.
            act_labels = tf.cast(
                tf.less(tf.abs(ph_q_values - best_qv), 0.01), 'float32')
            # act_labels = tf.cast(tf.equal(ph_q_values, best_qv), 'float32')
            label_sum = tf.reduce_sum(act_labels, axis=-1, keep_dims=True)
            act_label_dist = act_labels / label_sum
            # zero out disabled or dead-end actions!
            dead_end_value = to_local(
                problem_service.get_ssipp_dead_end_value())
            act_label_dist *= tf.cast(act_labels <= dead_end_value, 'float32')
            # XXX: this will obviously break if we have softmax; it'll spend
            # heaps of time trying to get all labels to be equal, and still
            # have (nonsense) nonzero loss afterwards :(
            xent = tf.reduce_mean(cross_entropy(act_dist, act_label_dist))
            loss_parts.append(('xent', xent))
        elif self.strategy == SupervisedObjective.MAX_ADVANTAGE:
            state_values = tf.reduce_min(ph_q_values, axis=-1)
            # is_nonzero = tf.greater(act_dist, 1e-4)
            # act_dist_nz = tf.where(is_nonzero, act_dist,
            #                        tf.ones_like(act_dist))
            # exp_q = act_dist_nz * (ph_q_values - state_values)
            exp_q = act_dist * ph_q_values
            exp_vs = tf.reduce_sum(exp_q, axis=-1)
            # state value is irrelevant to objective, but is included because
            # it ensures that zero loss = optimal policy
            q_loss = tf.reduce_mean(exp_vs - state_values)
            loss_parts.append(('qloss', q_loss))
            # XXX: need to look at whatever this is (and fix it if it's wrong)
            # if self.kl_coeff:
            #     assert self.kl_coeff > 0, \
            #         "negative entropy coefficient must be positive if supplied"
            #     is_nonzero = tf.equal(act_dist, 0.0)
            #     num_enabled = tf.reduce_sum(
            #         tf.cast(is_nonzero, tf.float32), axis=1)
            #     # clip so that really tiny values don't make our loss balloon!
            #     act_dist_clip = tf.clip_by_value(act_dist, 1e-10, 1.0)
            #     # also change all the zero values to ones, so that they count
            #     # as zero in summation below
            #     act_dist_clamp = tf.where(is_nonzero, act_dist_clip,
            #                               tf.ones_like(act_dist))
            #     xent = -tf.reduce_sum(
            #         tf.log(act_dist_clamp), axis=1) / num_enabled
            #     kl_div = -tf.log(num_enabled) + xent
            #     scale_kl_div = self.kl_coeff * tf.reduce_mean(kl_div)
            #     loss_parts.append(('scale-kld', scale_kl_div))
            #
            #     batch_neg_entropy = tf.reduce_sum(
            #         act_dist * tf.log(act_dist_clamp), axis=-1)
            #     # we allow drift of this many bits from uniform; otherwise,
            #     # apply entropy loss!
            #     num_enabled = tf.reduce_sum(
            #         tf.cast(act_dist > 1e-10, tf.float32), axis=1)
            #     allowed_bits = num_enabled - 1.5
            #     uniform_bits = tf.log(num_enabled) / tf.log(2.0)
            #     min_neg_entropy = -uniform_bits + allowed_bits
            #     batch_neg_ent_clip = tf.clip_by_value(batch_neg_entropy,
            #                                           min_neg_entropy, 0)
            #     batch_neg_ent_clip += min_neg_entropy
            #     # we want to maximise entropy, kinda
            #     ent_reg = self.neg_ent_coeff * tf.reduce_mean(
            #         batch_neg_ent_clip)
            #     loss_parts.append(('entreg', ent_reg))
        else:
            raise ValueError("Unknown strategy %s" % self.strategy)

        # regularisation
        # TODO: make this configurable!
        weights = self.weight_manager.all_weights
        l2_reg = 0.0 * sum(tf.nn.l2_loss(w) for w in weights)  # XXX disabled for monster
        loss_parts.append(('l2reg', l2_reg))

        loss = sum(p[1] for p in loss_parts)

        return ph_obs_var, ph_q_values, loss, loss_parts

    @can_profile
    def _init_tf(self):
        """Do setup necessary for network (e.g. initialising weights)."""
        assert not self.tf_init_done, \
            "this class is not designed to be initialised twice"
        self.sess = tf.get_default_session()
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.lr)

        # maps problem names to (obs var, q-value var) tuples
        self.obs_qv_inputs = {}
        losses = []
        loss_parts = None
        batch_sizes = []
        for problem in self.problems:
            this_obs_var, this_q_values, this_loss, this_loss_parts \
                = self._instantiate_net(problem)
            self.obs_qv_inputs[problem.name] = (this_obs_var, this_q_values)
            this_batch_size = tf.shape(this_obs_var)[0]
            losses.append(this_loss)
            batch_sizes.append(tf.cast(this_batch_size, tf.float32))
            if loss_parts is None:
                loss_parts = this_loss_parts
            else:
                # we care about these parts because we want to display them to
                # the user (e.g. how much of my loss is L2 regularisation
                # loss?)
                assert len(loss_parts) == len(this_loss_parts), \
                    'diff. loss breakdown for diff. probs. (%s vs %s)' \
                    % (loss_parts, this_loss_parts)
                # sum up all the parts
                new_loss_parts = []
                for old_part, new_part in zip(loss_parts, this_loss_parts):
                    assert old_part[0] == new_part[0], \
                        "names (%s vs. %s) don't match" % (old_part[0],
                                                           new_part[0])
                    to_add = new_part[1] * tf.cast(this_batch_size, tf.float32)
                    new_loss_parts.append((old_part[0], old_part[1] + to_add))
                loss_parts = new_loss_parts
        self.op_loss \
            = sum(l * s for l, s in zip(losses, batch_sizes)) \
            / sum(batch_sizes)
        # this is actually a list of (name, symbolic representation) pairs for
        # components of the loss
        self.loss_part_ops = [(name, value / sum(batch_sizes))
                              for name, value in loss_parts]

        # Next bit hairy because we want combined grads (and also want to split
        # them out for TensorBoard to look at). Really this is similar to
        # self.op_train = self.optimiser.minimize(loss).
        params = self.weight_manager.all_weights
        # do a check that set(params) is the same as
        param_set = set(params)
        for problem in self.problems:
            their_param_set = set(problem.policy.get_params(trainable=True))
            assert their_param_set == param_set, \
                "policy for %s has weird params" % problem.name

        grads_and_vars = self.optimiser.compute_gradients(
            self.op_loss, var_list=params)
        # see https://stackoverflow.com/a/43486487 for gradient clipping
        gradients, variables = zip(*grads_and_vars)
        gradients = list(gradients)
        # for grad, var in grads_and_vars:
        #     gradients[0] = tf.Print(gradients[0], [tf.norm(grad), tf.norm(var)], 'grad/var norm for %s:' % var.name)
        grads_and_vars = zip(gradients, variables)
        self.op_train = self.optimiser.apply_gradients(
            grads_and_vars=grads_and_vars)
        for g, v in grads_and_vars:
            tf.summary.histogram(
                'weight-grads/' + v.name, g, collections=['sl-hists'])
            for slot in self.optimiser.get_slot_names():
                slot_var = self.optimiser.get_slot(v, slot)
                if slot_var is not None:
                    dest_name = 'slots-' + slot + '/' + v.name
                    tf.summary.histogram(
                        dest_name, slot_var, collections=['sl-hists'])

        # "weights" is probably set by some code somewhere deep in RLLab
        # TODO: this is probably not the best idea. Maybe do weight hist stuff
        # *here*?
        weight_op = tf.summary.merge_all('weights')
        # 'summaries_f_prob' (for activations) is set up in
        # CategoricalMLPPolicy.__init__. Again I stuck it deep in RLLab because
        # I'm an idiot.
        act_op = tf.summary.merge_all('sl-activations')
        tf.summary.merge([act_op, weight_op], collections=['sl-hists'])
        self.op_summary = tf.summary.merge_all('sl-hists')

        # tensorboard ops
        self._log_ops = {}

        self.sess.run(tf.global_variables_initializer())

        self.tf_init_done = True

    def _get_log_op(self, name: str) -> Tuple[tf.Operation, tf.Tensor]:
        """Get trivial TF op to register scalar quantity (e.g. loss) with
        TB. Useful part is _log_op_value(name, some value)!"""
        collection = 'sl-diagnostics'
        if name not in self._log_ops:
            # shape=() means "scalar"; (1,) doesn't work for scalars because TF
            # won't upcast
            new_placeholder = tf.placeholder(
                tf.float32, shape=(), name=name + '_in')
            new_summary = tf.summary.scalar(
                name, new_placeholder, collections=[collection])
            # insert into dictionary so we don't have to recreate
            self._log_ops[name] = (new_summary, new_placeholder)
        summary_op, placeholder = self._log_ops[name]
        return summary_op, placeholder

    def _log_op_value(self, name: str, value: Any) -> None:
        """Actually add a float value to the TB log w/ given name."""
        summary_op, placeholder = self._get_log_op(name)
        sess = tf.get_default_session()
        result = sess.run(summary_op, {placeholder: value})
        self.summary_writer.add_summary(result, self.batches_seen)

    @can_profile
    def _make_batches(self, n_batches: int) -> Iterable[Dict[Any, Any]]:
        """Make a given number of batches for each problem."""
        batch_iters = []
        for problem in self.problems:
            service = problem.problem_service
            it = service.batch_iter(self.batch_size_per_problem, n_batches)
            batch_iters.append(it)
        combined = zip(*batch_iters)
        # yield a complete feed dict
        for combined_batch in combined:
            assert len(combined_batch) == len(self.problems)
            yield_val = {}
            for problem, batch in zip(self.problems, combined_batch):
                ph_obs_var, ph_q_values = self.obs_qv_inputs[problem.name]
                obs_tensor, qv_tensor = to_local(batch)
                yield_val[ph_obs_var] = obs_tensor
                yield_val[ph_q_values] = qv_tensor
            yield yield_val

    @can_profile
    def _optimise(self, n_batches: int) -> float:
        """Do SGD against states in replay pool."""
        all_batches_iter = self._make_batches(n_batches)
        tr = tqdm.tqdm(all_batches_iter, desc='batch', total=n_batches)
        losses = []
        part_names, part_ops = zip(*self.loss_part_ops)
        part_ops = list(part_ops)
        for batch_num, feed_dict in enumerate(tr):
            if (self.batches_seen % 50) == 0:
                run_result = self.sess.run(
                    [self.op_loss, self.op_train, self.op_summary] + part_ops,
                    feed_dict=feed_dict)
                loss, _, summ = run_result[:3]
                part_losses = run_result[3:]
                self.summary_writer.add_summary(summ, self.batches_seen)
            else:
                run_result = self.sess.run(
                    [self.op_loss, self.op_train] + part_ops,
                    feed_dict=feed_dict)
                loss = run_result[0]
                part_losses = run_result[2:]
            tr.set_postfix(loss=loss)
            losses.append(loss)
            self.batches_seen += 1
            self._log_op_value('train-loss', loss)
            assert len(part_names) == len(part_losses)
            for part_loss_name, part_loss in zip(part_names, part_losses):
                self._log_op_value('loss-%s' % part_loss_name, part_loss)
        return np.mean(losses)
