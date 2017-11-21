"""Interface code for rllab. Mainly handles interaction with mdpsim & hard
things like action masking."""

import abc
from warnings import warn

import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces import Discrete, Box
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.envs.base import TfEnv
import numpy as np
from rllab.envs.normalized_env import normalize

import ssipp_interface
from ssipp_interface import format_state_for_ssipp
from tf_utils import masked_softmax


def create_environment(problem_meta,
                       planner_extensions,
                       heuristic_name=None,
                       use_lm_cuts=False,
                       dump_table_path=None,
                       dump_table_interval=None):
    data_gens = [
        ActionEnabledGenerator(), RelaxedDeadendDetector(
            planner_extensions,
            dump_path=dump_table_path,
            dump_interval=dump_table_interval)
    ]
    if heuristic_name is not None:
        print('Creating heuristic feature generator (h=%s)' % heuristic_name)
        heur_gen = HeuristicDataGenerator(planner_extensions, heuristic_name)
        data_gens.append(heur_gen)
    if use_lm_cuts:
        print('Creating lm-cut heuristic feature generator')
        lm_cut_gen = LMCutDataGenerator(planner_extensions)
        data_gens.append(lm_cut_gen)
    # this is the actual ProblemEnvironment which I wrote for MDPSim
    raw_env = ProblemEnv(
        planner_extensions.mdpsim_problem,
        [p.unique_ident for p in problem_meta.bound_props_ordered],
        [a.unique_ident for a in problem_meta.bound_acts_ordered],
        data_gens=data_gens)
    # this one is wrapped to return actual vectors in response to interaction
    flat_env = TfEnv(
        normalize(FlattenObsWrapper(raw_env), normalize_obs=False))
    return raw_env, flat_env


class FPGObservation:
    def __init__(self, props_true, enabled_actions):
        # XXX: can I rely on a stable ordering for these? Not immediately
        # obvious. At the same time, I might break things if I apply my own
        # ordering.
        self.props_true = tuple(props_true)
        self._props_true_str \
            = tuple((p.identifier, v) for p, v in self.props_true)

        self.enabled_actions = tuple(enabled_actions)

        self._enabled_actions_str \
            = tuple((a.identifier, v) for a, v in self.enabled_actions)

    def __repr__(self):
        return 'FPGObservation(%r, %r)' % (self.props_true,
                                           self.enabled_actions)

    @property
    def _ident(self):
        return (self._props_true_str, self._enabled_actions_str)

    def __hash__(self):
        return hash(self._ident)

    def __eq__(self, other):
        if not isinstance(other, FPGObservation):
            return NotImplemented
        return self._ident == other._ident


class ProblemEnv(Env):
    def __init__(self, problem, prop_order, act_order, data_gens=tuple()):
        self.problem = problem
        self.state = problem.init_state()
        self.data_gens = list(data_gens)
        self.prop_order = prop_order
        self.act_order = act_order
        self.act_ident_to_act = {
            a.identifier: a
            for a in problem.ground_actions
        }
        self.reset()

    def step(self, action_id):
        assert isinstance(action_id, int), \
            "Action must be integer, but is %s" % type(action_id)

        action_name = self.act_order[action_id]
        # the != True is defensive; I found before that the array just had
        # true-ish values which really meant "disabled"!
        action = self.act_ident_to_act[action_name]
        applicable = self.problem.applicable(self.state, action)
        assert isinstance(applicable, bool)
        assert applicable == self.enabled_acts[action_id], \
            "problem says applicable = %r, contradicting " \
            "enabled_acts[id] = %r" % (applicable, self.enabled_acts)
        if not applicable:
            tot_enabled = sum(self.enabled_acts)
            tot_avail = len(self.enabled_acts)
            warn('Selected disabled action #%d, %s; %d/%d available' %
                 (action_id, action_name, tot_enabled, tot_avail))
        else:
            self.state = self.problem.apply(self.state, action)

        props = self.problem.prop_truth_mask(self.state)
        actions_enabled = self.problem.act_applicable_mask(self.state)

        # observe
        obs = FPGObservation(props_true=props, enabled_actions=actions_enabled)
        self.enabled_acts = self.observation_space.make_act_bools(obs)

        # we're done if we reach goal or run out of actions
        done = self.state.goal() \
            or not any(self.enabled_acts) \
            or self.is_recognised_dead_end(props)

        # Reward shaping: I disabled this because it was really misleading on
        # blocksworld, and doesn't seem to help much on anything else.
        # reward (0.1/1 instead of 100/1000)
        # rew = 0.1 * (step_data.progress - self._progress)
        rew = 0
        if self.state.goal():
            rew += 1

        # extra info
        info = {
            'goal_reached': self.state.goal(),
            # XXX: I don't currently calculate the real cost; must sort that
            # out!
            'step_cost': 1,
        }

        return obs, rew, done, info

    def reset(self):
        self.state = self.problem.init_state()
        obs = self.problem.prop_truth_mask(self.state)
        act_mask = self.problem.act_applicable_mask(self.state)
        self.enabled_acts = [v for _, v in act_mask]
        rv = FPGObservation(obs, act_mask)
        self.enabled_acts = self.observation_space.make_act_bools(rv)
        return rv

    def is_recognised_dead_end(self, all_prop_truth):
        # check whether self.data_gens indicate this is a dead end
        for data_gen in self.data_gens:
            if data_gen.is_dead_end(all_prop_truth):
                return True
        return False

    def action_name(self, action_id):
        if 0 <= action_id < len(self.act_order):
            return self.act_order[action_id]

    @property
    def observation_dim(self):
        extra_dims = sum(dg.extra_dim for dg in self.data_gens)
        nprops = self.problem.num_props
        nacts = self.problem.num_actions
        return nprops + (1 + extra_dims) * nacts

    @property
    def action_space(self):
        return Discrete(self.problem.num_actions)

    @property
    def observation_space(self):
        return FPGObservationSpace(self.problem, self.prop_order,
                                   self.act_order, self.data_gens)


class FlattenObsWrapper(ProxyEnv):
    """Environment wrapper which takes care of flattening/unflattening
    observations. This is necessary for interfacing with TF because RLLab's TF
    code doesn't flatten/unflatten observations like the Theano code does."""

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        return self._convert_obs(obs), rew, done, info

    def reset(self):
        obs = self._wrapped_env.reset()
        return self._convert_obs(obs)

    def _convert_obs(self, obs):
        return self._wrapped_env.observation_space.flatten(obs)

    @property
    def observation_space(self):
        wrap_space = self._wrapped_env.observation_space
        assert isinstance(wrap_space, FPGObservationSpace)
        return Box(wrap_space.low.min(),
                   wrap_space.high.max(), wrap_space.shape)


def _mask_to_str_dict(mask):
    rv = {}
    for real_item, value in mask:
        ident = real_item.identifier
        if ident in rv:
            raise ValueError('Identifier "%s" appears twice in mask!' % ident)
        rv[ident] = value
    return rv


class ActionDataGenerator(abc.ABC):
    """ABC for things which stuff extra data into the input of each action."""

    @abc.abstractmethod
    def get_extra_data(self, state_pairs, act_order, acts_enabled):
        """Get extra data from a state (expressed as list of (Proposition,
        truth value) pairs)"""
        pass

    @abc.abstractproperty
    def extra_dim(self):
        """Elements of extra data added per action."""
        pass

    def is_dead_end(self, all_props):
        """Check whether this is a dead end (return False if unsure)."""
        return False


class ActionEnabledGenerator(ActionDataGenerator):
    extra_dim = 1

    def get_extra_data(self, all_props, act_order, acts_enabled):
        out_vec = np.zeros((len(act_order), self.extra_dim))
        enable_set = set(a.identifier for a, e in acts_enabled if e is True)
        for idx, act_name in enumerate(act_order):
            if act_name in enable_set:
                out_vec[idx] = 1
        return out_vec


class SSiPPDataGenerator(ActionDataGenerator):
    """Basic class for generators which use SSiPP"""

    def __init__(self, mod_sandbox):
        self.mod_sandbox = mod_sandbox
        self.ssipp_problem = self.mod_sandbox.ssipp_problem


class RelaxedDeadendDetector(SSiPPDataGenerator):
    """Checks for dead ends in delete relaxation. No extra_dim because it only
    looks for dead ends."""
    extra_dim = 0

    def __init__(self, mod_sandbox, dump_path=None, dump_interval=50000):
        super().__init__(mod_sandbox)
        self.evaluator = ssipp_interface.Evaluator(self.mod_sandbox, "h-max")
        # dump_interval allows us to dump
        # XXX: this is stupid and hacky; should ideally have another callback
        # for doing debug stuff in these data generators
        self.dump_interval = dump_interval
        self.dump_path = dump_path
        self.dump_wait = 0

    def get_extra_data(self, all_props, act_order, acts_enabled):
        return np.zeros((len(act_order), self.extra_dim))

    def is_dead_end(self, all_props):
        state_string = format_state_for_ssipp(all_props)
        state_value = self.evaluator.eval_state(state_string)
        is_dead_end = state_value >= self.mod_sandbox.ssipp_dead_end_value
        if self.dump_path is not None and self.dump_interval:
            # write the heuristic table out to supplied path
            self.dump_wait += 1
            if self.dump_wait >= self.dump_interval:
                print("Dumping table to %s" % self.dump_path)
                self.evaluator.dump_table(self.dump_path)
                self.dump_wait = 0
        return is_dead_end


class LMCutDataGenerator(SSiPPDataGenerator):
    """Adds 'this is in a disjunctive cut'-type flags to propositions."""
    extra_dim = 2
    IN_ANY_CUT = 0
    IN_SINGLETON_CUT = 1

    def __init__(self, *args):
        super().__init__(*args)
        self.cutter = ssipp_interface.Cutter(self.mod_sandbox)

    def get_extra_data(self, all_props, act_order, acts_enabled):
        out_vec = np.zeros((len(act_order), self.extra_dim))
        state_string = format_state_for_ssipp(all_props)
        cuts = self.cutter.get_action_cuts(state_string)
        in_unary_cut = set()
        in_any_cut = set()
        for cut in cuts:
            if len(cut) == 1:
                in_unary_cut.update(cut)
            if len(cut) >= 1:
                # all actions in cuts (unary cuts or not) end up here
                in_any_cut.update(cut)
        assert (in_unary_cut | in_any_cut) <= set(act_order), \
            "there are some things in cuts that aren't in action set"
        for idx, act_name in enumerate(act_order):
            if act_name in in_unary_cut:
                out_vec[idx][self.IN_SINGLETON_CUT] = 1
            if act_name in in_any_cut:
                out_vec[idx][self.IN_ANY_CUT] = 1
        return out_vec


class HeuristicDataGenerator(SSiPPDataGenerator):
    """Will create some trivial features based on successors yielded by an
    action. Concretely, produces an indicator vector with the following
    layout:
        0: action disabled?
        1: best outcome decreases heuristic?
        2: best outcome increases heuristic?
        3: best outcome keeps heuristic same?
    Can probably fiddle around with this quite a bit (e.g. to add delta in dead
    end probabilities, etc.).
    """
    extra_dim = 4
    DISABLED = 0
    DECREASE = 1
    INCREASE = 2
    SAME = 3

    def __init__(self, mod_sandbox, heuristic_name):
        super().__init__(mod_sandbox)
        self.heuristic_name = heuristic_name
        self.evaluator = ssipp_interface.Evaluator(self.mod_sandbox,
                                                   heuristic_name)

    def get_extra_data(self, all_props, act_order, acts_enabled):
        out_vec = np.zeros((len(act_order), self.extra_dim))
        state_string = format_state_for_ssipp(all_props)
        state_value = self.evaluator.eval_state(state_string)
        enable_set = set(a.identifier for a, e in acts_enabled if e is True)
        for idx, act_name in enumerate(act_order):
            if act_name not in enable_set:
                out_vec[idx][self.DISABLED] = 1
            else:
                succ_probs_vals = self.evaluator.succ_probs_vals(state_string,
                                                                 act_name)
                best_outcome = min(val for _, val in succ_probs_vals)
                if best_outcome < state_value:
                    out_vec[idx][self.DECREASE] = 1
                elif best_outcome == state_value:
                    out_vec[idx][self.SAME] = 1
                else:  # greater
                    assert best_outcome > state_value
                    out_vec[idx][self.INCREASE] = 1
        return out_vec

    def is_dead_end(self, all_props):
        state_string = format_state_for_ssipp(all_props)
        state_value = self.evaluator.eval_state(state_string)
        return state_value >= ssipp_interface.dead_end_value()


class FPGObservationSpace(Box):
    """Observation space for MDPSim environments. Will flatten down
    observations into a single numerical vector containing predicate values,
    action mask, etc; it's up to the code to *unflatten* it back into something
    meaningful."""

    def __init__(self, mdpsim_prob, prop_order, act_order, data_gens=tuple()):
        self._data_gens = list(data_gens)
        self._extra_dim = sum(dg.extra_dim for dg in self._data_gens)
        if self._data_gens:
            assert self._extra_dim > 0, \
                "data generators don't add anything, probable bug"
        dims = (1 + self._extra_dim) * mdpsim_prob.num_actions \
            + mdpsim_prob.num_props

        super().__init__(low=0, high=1, shape=(dims, ))

        self._problem = mdpsim_prob

        self._prop_dim = self._problem.num_props
        self._prop_order = prop_order
        assert self._prop_dim == len(self._prop_order)

        self._act_dim = self._problem.num_actions
        self._act_order = act_order
        assert self._act_dim == len(self._act_order)

    def _conv_bools(self, obs):
        assert isinstance(obs, list)
        assert all(isinstance(e, (bool, np.bool_)) for e in obs)
        return np.array(obs, dtype='float32')

    def flatten(self, x):
        assert isinstance(x, FPGObservation)

        prop_dict = _mask_to_str_dict(x.props_true)
        assert len(prop_dict) == self._prop_dim
        props = [prop_dict[pi] for pi in self._prop_order]
        props_conv = self._conv_bools(props)

        act_mask_conv = self._conv_bools(self.make_act_bools(x))

        if self._extra_dim > 0:
            # mats are num actions * num features (where second dimension
            # depends on the data generator)
            extra_mats = [
                dg.get_extra_data(x.props_true, self._act_order,
                                  x.enabled_actions) for dg in self._data_gens
            ]
            # concat along features (size varies), then drop to vector
            extra_data = np.concatenate(extra_mats, axis=1).flatten()
            # action mask first, extra data second, observation third
            # (important for policy!)
            rv = np.concatenate((act_mask_conv, extra_data, props_conv))
        else:
            rv = np.concatenate(act_mask_conv, props_conv)

        # sanity
        # no reverse check in .unflatten() because .flatten() is bloody
        # expensive when you have fancy data gens
        # TODO: cache this, actually
        assert self.unflatten(rv) == x, (x, self.unflatten(rv))

        return rv

    def make_act_bools(self, x):
        act_dict = _mask_to_str_dict(x.enabled_actions)
        assert len(act_dict) == self._act_dim, \
            "expected %d actions, got %d" % (self._act_dim, len(act_dict))
        act_mask = [act_dict[ai] for ai in self._act_order]
        return act_mask

    def unflatten(self, x):
        assert x.ndim == 1, x.shape
        assert x.size == self._act_dim * (1 + self._extra_dim) \
            + self._prop_dim, x.shape

        prob_spec = self._problem

        # props last
        props_mask = x[-self._prop_dim:] != 0
        assert props_mask.shape == (self._prop_dim, ), props_mask.shape
        props = zip(prob_spec.propositions, props_mask)

        # acts first
        acts_mask = x[:self._act_dim] != 0
        assert acts_mask.shape == (self._act_dim, ), acts_mask.shape
        acts = zip(prob_spec.ground_actions, acts_mask)

        rv = FPGObservation(props_true=props, enabled_actions=acts)

        return rv

    def flatten_n(self, xs):
        return np.stack(list(map(self.flatten, xs)), axis=0)

    def unflatten_n(self, xs):
        return list(map(xs, self.unflatten))

    def __repr__(self):
        return 'FPGObservationSpace({})'.format(self.mdpsim_env)


# XXX: this code was copied from RLLab because it's too hard to extend :P


class MaskedMLP(LayersPowered, Serializable):
    def __init__(
            self,
            name,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            hidden_W_init=L.XavierUniformInitializer(),
            hidden_b_init=tf.zeros_initializer(),
            output_W_init=L.XavierUniformInitializer(),
            output_b_init=tf.zeros_initializer(),
            input_var=None,
            input_layer=None,
            input_shape=None,
            batch_normalization=False,
            weight_normalization=False, ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            if input_layer is None:
                assert input_shape is not None, \
                    "input_layer or input_shape must be supplied"
                l_in = L.InputLayer(
                    shape=(None, ) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization)
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out_raw = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization)
            if batch_normalization:
                l_out_raw = L.batch_norm(l_out_raw)
            self._layers.append(l_out_raw)

            # mask assumed to occupy first output_dim elements
            def mask_op(X):
                return X[..., :output_dim]

            def mask_shape_op(old_shape):
                return old_shape[:-1] + (output_dim, )

            mask = L.OpLayer(l_in, mask_op, shape_op=mask_shape_op)
            self._layers.append(mask)
            l_out = L.OpLayer(l_out_raw, masked_softmax, extras=[mask])
            self._layers.append(l_out)

            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output


class PrintShapeLayer(L.Layer):
    """Prints shape of its sole incoming with some nice message."""

    def __init__(self, incoming, message, **kwargs):
        super(PrintShapeLayer, self).__init__(incoming, **kwargs)
        self.message = message

    def get_output_for(self, input, **kwargs):
        return tf.Print(input, [tf.shape(input)], message=self.message)

    def get_output_shape_for(self, input_shape):
        return input_shape


class MatchBatchLayer(L.MergeLayer):
    def __init__(self, passthrough, other, message=None, **kwargs):
        super(MatchBatchLayer, self).__init__([passthrough, other], **kwargs)
        if message is None:
            message = 'sizes of passthrough layer %s (%s) and other layer ' \
                      '%s (%s) do not match' % (passthrough.name, passthrough,
                                                other.name, other)
        self.message = message
        if self.name is None and passthrough.name is not None:
            self.name = 'matchbatch-for-%s' % passthrough.name

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        to_return, to_match = inputs
        match_bs = tf.assert_equal(
            tf.shape(to_return)[0],
            tf.shape(to_match)[0],
            message=self.message)
        with tf.control_dependencies([match_bs]):
            return tf.identity(to_return, name='passthrough')


def make_masked_mlp(name,
                    observation_dim,
                    act_dim,
                    hidden_sizes=(32, 32),
                    hidden_nonlinearity=tf.nn.tanh):
    assert observation_dim > act_dim, \
        "observations should be prefixed by actions, but sizes don't check out"

    return MaskedMLP(
        name,
        act_dim,
        hidden_sizes,
        hidden_nonlinearity,
        input_shape=(observation_dim, ))


def det_sampler(weights):
    """Like <action space>.weighted_sample, but always chooses max prob
    action."""
    return int(np.argmax(weights))


def unwrap_env(env):
    """Remove all layers of environment wrapper."""
    while hasattr(env, '_wrapped_env'):
        env = env._wrapped_env
    return env
