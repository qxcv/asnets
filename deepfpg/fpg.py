#!/usr/bin/env python3

import argparse
import atexit
from contextlib import ExitStack
from json import dump
from os import makedirs, path
from random import randint
import sys
from time import time
from typing import Any, Dict, Set  # noqa

import joblib
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.normalized_env import normalize
from rllab.misc import logger
import rpyc
# from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import \
    CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.npo import NPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.optimizers.first_order_optimizer import \
    FirstOrderOptimizer
import tensorflow as tf
# for some reason "import tensorflow.python.debug" doesn't work
from tensorflow.python import debug as tfdbg

# from rllab_interface import ProblemEnv, make_masked_mlp, FlattenObsWrapper, \
#     HeuristicDataGenerator, ActionEnabledGenerator, RelaxedDeadendDetector, \
#     LMCutDataGenerator, det_sampler
from rllab_interface import make_masked_mlp, det_sampler
from models import PropNetworkWeights, PropNetwork
from supervised import SupervisedTrainer, SupervisedObjective, \
    ProblemServiceConfig, RemoteEnv
from multiprob import ProblemServer, to_local


def run_trial(policy,
              problem_server,
              limit=1000,
              det_sample=False,
              show_time=False):
    """Run policy on problem. Returns (cost, path), where cost may be None if
    goal not reached before horizon."""
    env = RemoteEnv(problem_server)
    obs = env.reset()
    # total cost of this run
    cost = 0
    path = []
    if det_sample:
        bak_sample = policy.action_space.weighted_sample
        policy.action_space.weighted_sample = det_sampler
    try:
        for action_num in range(1, limit):
            if show_time:
                start = time()
            action, _ = policy.get_action(obs)
            if show_time:
                elapsed = time() - start
                print('get_action took %fs' % elapsed)
            path.append(env.action_name(action))
            obs, reward, done, step_info = env.step(action)
            cost += step_info['step_cost']
            if step_info['goal_reached']:
                path.append('GOAL! :D')
                return cost, True, path
            # we can run out of time or run out of actions to take
            if done:
                break
        path.append('FAIL! D:')
    finally:
        if det_sample:
            policy.action_space.weighted_sample = bak_sample
    return cost, False, path


def run_trials(policy,
               problem_server,
               trials,
               limit=1000,
               det_sample=False,
               show_time=False):
    all_exec_times = []
    all_costs = []
    all_goal_reached = []
    paths = []
    for i in range(trials):
        start = time()
        cost, goal_reached, path = run_trial(policy, problem_server, limit,
                                             det_sample, show_time)
        elapsed = time() - start
        paths.append(path)
        all_exec_times.append(elapsed)
        all_costs.append(cost)
        all_goal_reached.append(goal_reached)
    meta_dict = {
        'turn_limit': limit,
        'trials': trials,
        'all_goal_reached': all_goal_reached,
        'all_exec_times': all_exec_times,
        'all_costs': all_costs,
    }
    return meta_dict, paths


_opt_dict = {}  # type: Dict[str, Any]


def opt(entry_name, extra_opts={}):
    """Adds an optimiser to the optimiser registry, allowing it to be chosen
    from command line."""

    def decorator(f):
        assert entry_name not in _opt_dict
        _opt_dict[entry_name] = f
        f.optimiser_name = entry_name
        f.extra_opts = extra_opts
        return f

    return decorator


@opt('trpo')
def opt_trpo(env, baseline, policy, **kwargs):
    return TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        discount=0.99,
        step_size=0.001,
        batch_size=4000,
        n_itr=int(1e9),
        **kwargs)


@opt('npo')
def opt_npo(env, baseline, policy, **kwargs):
    # Seems to be normal NPO
    # TODO: try implementing the "proximal policy optimisation" described in
    # Abbeel & Schulman's NIPS '16 talk. I think that's just SGD on (f + a *
    # KL), where a increases if KL gets too large.
    return NPO(env=env,
               policy=policy,
               baseline=baseline,
               discount=0.99,
               step_size=0.001,
               batch_size=4000,
               n_itr=int(1e9),
               optimizer_args=dict(name='lbfgs_scope'),
               **kwargs)


@opt('vpg', {'learning_rate': float, 'batch_size': int})
def opt_vpg(env,
            baseline,
            policy,
            learning_rate=1e-5,
            batch_size=4000,
            **kwargs):
    # no idea what batch size, learning rate, etc. should be
    optimiser = FirstOrderOptimizer(
        tf_optimizer_cls=tf.train.AdamOptimizer,
        tf_optimizer_args=dict(learning_rate=learning_rate),
        # batch_size actually gets passed to BatchPolopt (parent of VPG)
        # instead of TF optimiser (makes sense, I guess)
        batch_size=None,
        max_epochs=1)
    return VPG(env=env,
               policy=policy,
               baseline=baseline,
               n_itr=int(1e9),
               optimizer=optimiser,
               batch_size=batch_size,
               **kwargs)


def argparse_optimiser(str_arg):
    try:
        return _opt_dict[str_arg]
    except KeyError:
        raise argparse.ArgumentError(
            'Unknown optimiser "%s". Known optimisers: %s' %
            (str_arg, ', '.join(_opt_dict.keys())))


def unique_name(args, digits=6):
    rand_num = randint(1, (1 << (4 * (digits + 1)) - 1))
    suffix = '{:x}'.format(rand_num).zfill(digits)
    if args.timeout is None:
        time_str = 'inf'
    else:
        time_str = '%d' % round(args.timeout)
    mo_str = ','.join('%s=%s' % (k, v) for k, v in args.model_opts.items())
    if args.problems:
        all_probs_comma = ','.join(args.problems)
        if len(all_probs_comma) > 50:
            all_probs_comma = all_probs_comma[:47] + '...'
        start = 'P[{}]'.format(all_probs_comma)
    else:
        names = []
        for pf in args.pddls:
            # remove directory path
            bn = path.basename(pf)
            pf_suffix = '.pddl'
            if bn.endswith(pf_suffix):
                # chop off extension
                bn = bn[:-len(pf_suffix)]
            if bn:
                names.append(bn)
        all_names_comma = ','.join(names)
        if len(all_names_comma) > 50:
            all_names_comma = all_names_comma[:47] + '...'
        start = 'P[%s]' % all_names_comma
    prefix = '{}-S[{},{},{}]-O[{}]-M[{}]-MO[{}]-T[{}]'.format(
        start, args.supervised_lr, args.supervised_bs,
        args.supervised_teacher_heur, args.optimiser.optimiser_name,
        args.model, mo_str, time_str)
    start_time_str = str(int(time() / 60 - 24881866)).zfill(8)
    return prefix + '-' + start_time_str + '-' + suffix


def opt_str(in_str):
    rv = {}
    for item in in_str.split(','):
        item = item.strip()
        if not item:
            continue
        name, value = item.split('=', 1)
        rv[name] = value
    return rv


parser = argparse.ArgumentParser(
    description='Trainer for Deep Factored Policy Gradient planner (DFPG).')
parser.add_argument(
    '--show-eval-time',
    action='store_true',
    default=False,
    help='print out times for each network evaluation during trials')
parser.add_argument(
    '-p',
    '--problem',
    dest='problems',
    action='append',
    help='name of problem to solve (can use this flag many times)')
parser.add_argument(
    '--opt-patience',
    type=int,
    default=10,
    help="if best observed undiscounted mean reward is >=1, *and* there has "
    "been no improvement for this many epochs, then we stop.")
parser.add_argument(
    '--supervised',
    action='store_true',
    default=False,
    help='train supervised instead of with RL')
parser.add_argument(
    '--supervised-lr',
    type=float,
    default=0.0005,
    help='learning rate for supervised learning')
parser.add_argument(
    '--supervised-bs',
    type=int,
    default=128,
    help='batch size for supervised learning')
parser.add_argument(
    '--supervised-kl-coeff',
    type=float,
    default=0.1,
    help='scale for KL(uniform||pi) term in supervised loss')
parser.add_argument(
    '--supervised-teacher-heur',
    default='lm-cut',
    choices=['lm-cut', 'h-add', 'h-max', 'simpleZero', 'smartZero'],
    help='heuristic to use for SSiPP teacher in supervised mode')
parser.add_argument(
    '--supervised-early-stop',
    type=int,
    default=12,
    help='halt after this many epochs with succ. rate >0.8 but no increase')
parser.add_argument(
    '-o',
    '--optimiser',
    default=opt_trpo,
    type=argparse_optimiser,
    help='name of optimiser to use')
parser.add_argument(
    '--dump-table-interval',
    default=50000,
    type=int,
    help="if a path is supplied, dead end checker's state table will be "
    "dumped periodically after this many iterations")
parser.add_argument(
    '--dump-table-path',
    default=None,
    help="path to which to dump dead end checker's state table "
    "(no dumping by default)")
parser.add_argument(
    '-A',
    '--optimiser-opts',
    default={},
    type=opt_str,
    help='additional arguments for optimiser')
parser.add_argument(
    '--resume-from', default=None, help='snapshot pickle to resume from')
parser.add_argument(
    '-t',
    '--timeout',
    type=float,
    default=None,
    help='maximum training time (disabled by default)')
parser.add_argument(
    '-m', '--model', choices=['simple', 'actprop'], default='simple')
parser.add_argument(
    '-O',
    '--model-opts',
    type=opt_str,
    default={},
    help='options for model (e.g. p1=v1,p2=v2,p3=v3)')
parser.add_argument(
    '--det-eval',
    action='store_true',
    default=False,
    help='use deterministic action selection for evaluation')
parser.add_argument(
    '-H',
    '--heuristic',
    type=str,
    default=None,
    help='SSiPP heuristic to give to DFPG')
parser.add_argument(
    '--no-use-lm-cuts',
    dest='use_lm_cuts',
    default=True,
    action='store_false',
    help="don't add flags indicating which actions are in lm-cut cuts")
parser.add_argument(
    '-R', '--rounds-eval', type=int, default=100, help='number of eval rounds')
parser.add_argument(
    '-L', '--limit-turns', type=int, default=100, help='max turns per round')
parser.add_argument(
    '-e', '--expt-dir', default=None, help='path to store experiments in')
parser.add_argument(
    '--debug',
    default=False,
    action='store_true',
    help='enable tensorflow debugger')
parser.add_argument(
    '--no-train',
    default=False,
    action='store_true',
    help="don't train, just evaluate")
parser.add_argument(
    'pddls', nargs='+', help='paths to PDDL domain/problem definitions')


def eval_single(args, policy, problem_server, unique_prefix, elapsed_time,
                iter_num, weight_manager, scratch_dir):
    # now we evaluate the learned policy
    print('Evaluating policy')
    trial_results, paths = run_trials(
        policy,
        problem_server,
        args.rounds_eval,
        limit=args.limit_turns,
        det_sample=args.det_eval,
        show_time=args.show_eval_time)

    print('Trial results:')
    print('\n'.join('%s: %s' % (k, v) for k, v in trial_results.items()))

    out_dict = {
        'no_train': args.no_train,
        'args_problems': args.problems,
        'problem': to_local(problem_server.service.get_current_problem_name()),
        'timeout': args.timeout,
        'optimiser': args.optimiser.optimiser_name,
        'model': args.model,
        'model_opts': args.model_opts,
        'all_args': sys.argv[1:],
        # TODO: possibly add this. Not sure whether it's worthwhile given
        # that the supposed "convergence" measure might be spurious (e.g.
        # what if it just spikes up in reward briefly?).
        # convergence_* refers to first iteration at which best score was
        # encountere
        # 'convergence_time': convergence_time,
        # 'convergence_iters': convergence_iter,
        # elapsed_* also includes time/iterations spent looking for better
        # results after converging
        'elapsed_opt_time': elapsed_time,
        'elapsed_opt_iters': iter_num,
        'trial_paths': paths
    }
    out_dict.update(trial_results)
    result_path = path.join(scratch_dir, 'results.json')
    with open(result_path, 'w') as fp:
        dump(out_dict, fp, indent=2)
    # also write out lists of actions taken during final trial
    # TODO: should also write out some randomly chosen paths during training
    # TODO: also write out probabilities of each action for at least some paths
    # (or some states), and maybe even real Q-values of actions (would be
    # helpful!)
    actions_path = path.join(scratch_dir, 'trial-paths.txt')
    with open(actions_path, 'w') as fp:
        for alist in paths:
            fp.write(' -> '.join(alist))
            fp.write('\n\n')


def main_rllab(args, unique_prefix, snapshot_dir):
    # data_gens = [
    #     ActionEnabledGenerator(), RelaxedDeadendDetector(
    #         module_sandbox,
    #         dump_path=args.dump_table_path,
    #         dump_interval=args.dump_table_interval)
    # ]
    # if args.heuristic is not None:
    #     print('Creating heuristic feature generator (h=%s)' % args.heuristic)
    #     heur_gen = HeuristicDataGenerator(module_sandbox, args.heuristic)
    #     data_gens.append(heur_gen)
    # if args.use_lm_cuts:
    #     print('Creating lm-cut heuristic feature generator')
    #     lm_cut_gen = LMCutDataGenerator(module_sandbox)
    #     data_gens.append(lm_cut_gen)

    # print('Getting problem metadata')
    # dom_meta = get_domain_meta(problem.domain)
    # prob_meta = get_problem_meta(problem, dom_meta)
    # penv = ProblemEnv(
    #     problem, [p.unique_ident for p in prob_meta.bound_props_ordered],
    #     [a.unique_ident for a in prob_meta.bound_acts_ordered],
    #     data_gens=data_gens)
    # env = TfEnv(normalize(FlattenObsWrapper(penv), normalize_obs=False))

    # known_mo = set()  # type: Set[str]
    # if args.model == 'simple':
    #     known_mo = {'hidden_size', 'num_layers'}
    #     mod = args.model_opts
    #     default_hidden_size = max([penv.observation_dim, 32])
    #     hidden_size = int(mod.get('hidden_size', default_hidden_size))
    #     num_layers = int(mod.get('num_layers', 2))
    #     print('Layer size: %d' % hidden_size)
    #     print('Number of layers: %d' % num_layers)
    #     # The dense policy network should have two hidden layers, each with
    #     # <obs dim> units (thereabouts, anyway)
    #     custom_network = make_masked_mlp('simple_masked_mlp',
    #                                      penv.observation_dim, penv.action_dim,
    #                                      (hidden_size, ) * num_layers)
    # elif args.model == 'actprop':
    #     known_mo = {'hidden_size', 'num_layers', 'dropout'}
    #     mod = args.model_opts
    #     hs = int(mod.get('hidden_size', 8))
    #     num_layers = int(mod.get('num_layers', 2))
    #     dropout = float(mod.get('dropout', 0.0))
    #     print('hidden_size: %d, num_layers: %d, dropout: %f' %
    #           (hs, num_layers, dropout))
    #     if args.resume_from:
    #         print('Reloading weight manager (resuming training)')
    #         weight_manager = joblib.load(args.resume_from)
    #     else:
    #         print('Creating new weight manager (not resuming)')
    #         # TODO: should save all network metadata, including heuristic
    #         # configuration
    #         extra_dim = sum([g.extra_dim for g in data_gens])
    #         weight_manager = PropNetworkWeights(
    #             dom_meta,
    #             hidden_sizes=[(hs, hs)] * num_layers,
    #             extra_dim=extra_dim)
    #     custom_network = PropNetwork(
    #         weight_manager, prob_meta, dropout=dropout)
    # else:
    #     raise ValueError('Unknown network type "%s"' % args.model)

    # # What if a model option wasn't used?
    # unknown_mo = args.model_opts.keys() - known_mo
    # if unknown_mo:
    #     print(
    #         'WARNING: model options not understood by "%s" network: %s' %
    #         (args.model, ', '.join(unknown_mo)),
    #         file=sys.stderr)

    # policy = CategoricalMLPPolicy(
    #     env_spec=env.spec, prob_network=custom_network, name='policy')
    # # TODO: try replacing linear baseline with a decision tree baseline
    # # (AdaBoost or something). Zero baseline is a little worse on TTW-1, and
    # # MLP is also fractionally worse in wall time IIRC.
    # baseline = LinearFeatureBaseline(env_spec=env.spec)
    # # baseline = GaussianMLPBaseline(env_spec=env.spec)
    # # baseline = ZeroBaseline(env_spec=env.spec)

    # summary_path = path.join('tensorboard', unique_prefix)
    # sample_writer = tf.summary.FileWriter(
    #     summary_path, graph=tf.get_default_graph())
    # # need to give CategoricalMLPPolicy the summary writer *after* creating the
    # # network itself, or the network will not show up in TB
    # policy.summary_writer = sample_writer

    # start = time()

    # with ExitStack() as stack:
    #     sess = tf.Session()
    #     stack.enter_context(sess)
    #     # need to keep session around for as long as we want to use the
    #     # network which we build
    #     if args.debug:
    #         print('Enabling TensorFlow debugger')
    #         sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    #         sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
    #         stack.enter_context(sess.as_default())

    #     if args.supervised and not args.no_train:
    #         print('Training supervised')
    #         # TODO: make all this stuff (batch size, number of epochs for
    #         # optimisation, etc.) configurable.

    #         service_config = ProblemServiceConfig(
    #             args.pddls, args.problem, use_lm_cuts=args.use_lm_cuts)
    #         problem_server = ProblemServer(service_config)
    #         problem_service = problem_server.service
    #         sup_trainer = SupervisedTrainer(
    #             environment=env,
    #             prob_meta=prob_meta,
    #             policy=policy,
    #             # strategy=SupervisedObjective.MAX_ADVANTAGE,
    #             strategy=SupervisedObjective.ANY_GOOD_ACTION,
    #             problem_service=problem_service,
    #             lr=0.001)
    #         train_gen = sup_trainer.train()
    #         best_rate = None
    #         iter_num = 0
    #         succ_rate, mean_loss, replay_size = next(train_gen)
    #         while True:
    #             keep_going = True
    #             succ_rate, mean_loss, replay_size = train_gen.send(keep_going)
    #             iter_num += 1
    #             if best_rate is None or succ_rate > best_rate:
    #                 best_rate = succ_rate
    #                 # snapshot!
    #                 snapshot_path = path.join(snapshot_dir,
    #                                           'snapshot_%d_%f.pkl' %
    #                                           (iter_num, succ_rate))
    #                 weight_manager.save(snapshot_path)
    #     elif not args.no_train:
    #         print('Training with RL')
    #         # process extra options for optimiser
    #         extra_opt_kwargs = {}
    #         for opt_k, opt_v in args.optimiser_opts.items():
    #             assert opt_k in args.optimiser.extra_opts, \
    #                 "%s does not accept option %s" \
    #                 % (args.optimiser.optimiser_name, opt_k)
    #             conv_func = args.optimiser.extra_opts[opt_k]
    #             extra_opt_kwargs[opt_k] = conv_func(opt_v)
    #         algo = args.optimiser(
    #             env,
    #             baseline,
    #             policy,
    #             max_path_length=args.limit_turns,
    #             # n_envs=1 is necessary to disable environment pickling
    #             sampler_args=dict(n_envs=1),
    #             summary_writer=sample_writer,
    #             **extra_opt_kwargs)

    #         # here's where we'll have paths which we had on the way to the goal
    #         # train_routes_save_path = path.join(args.logdir,
    #         #     unique_prefix + '-train-paths')
    #         # makedirs(train_routes_save_path, exist_ok=True)

    #         print('Initiating training run')
    #         past_best_score = 0.0
    #         past_best_iter = 0
    #         # TODO: factor this out into a new train_rl.py file, similar to
    #         # supervised.py
    #         assert tf.get_default_session() is sess
    #         rl_trainer = algo.train(sess=sess, async=True)
    #         try:
    #             iter_num, iter_params, iter_logs = next(rl_trainer)
    #             while True:
    #                 should_stop = False
    #                 must_snap = False

    #                 # this early stopping check is pretty hacky; oh well
    #                 score = float(iter_logs['AverageReturn'])
    #                 if score > past_best_score:
    #                     past_best_score = score
    #                     past_best_iter = iter_num
    #                     must_snap = True
    #                     losing_patience = score >= 1
    #                 if score < 1:
    #                     # If we drop below 1 then we have to wait until we
    #                     # exceed past_best_score before we stop due to lack
    #                     # of patience. This avoids situations where we get
    #                     # a good score briefly, but drop back to a horribly
    #                     # policy.
    #                     losing_patience = False
    #                 # we quit if we're currently scoring at least 1
    #                 # (~optimal) *and* haven't improved for a while
    #                 if losing_patience and score >= 1 and \
    #                         iter_num - past_best_iter > args.opt_patience:
    #                     print('Waited %d iters with no improvement, stopping' %
    #                           (iter_num - past_best_iter))
    #                     should_stop = True

    #                 # also make sure we have enough time left
    #                 elapsed_time = time() - start
    #                 if args.timeout and elapsed_time >= args.timeout:
    #                     print('Timed out after %.2fs, stopping' % elapsed_time)
    #                     should_stop = True

    #                 iter_num, iter_params, iter_logs \
    #                     = rl_trainer.send(should_stop)
    #                 assert tf.get_default_session() is sess

    #                 if args.model == 'actprop' and must_snap:
    #                     # we only save on improvement
    #                     snapshot_path = path.join(snapshot_dir,
    #                                               'snapshot_%d.pkl' % iter_num)
    #                     weight_manager.save(snapshot_path)
    #         except StopIteration:
    #             print('Got StopIteration from trainer')
    #     else:
    #         # need to fill up stats values with garbage :P
    #         elapsed_time = iter_num = None
    #         # normally trainers do this
    #         sess.run(tf.global_variables_initializer())

    #     # Now test!
    #     if weight_manager is not None:
    #         weight_manager.save(path.join(snapshot_dir, 'snapshot_final.pkl'))
    #     eval_single(args)
    raise NotImplementedError()


def make_policy(args,
                env_spec,
                dom_meta,
                prob_meta,
                dg_extra_dim=None,
                weight_manager=None):
    # size of input and output
    obs_dim = int(env_spec.observation_space.flat_dim)
    act_dim = int(env_spec.action_space.flat_dim)

    # can make normal FC MLP or an action/proposition network
    known_mo = set()  # type: Set[str]
    if args.model == 'simple':
        known_mo = {'hidden_size', 'num_layers'}
        mod = args.model_opts
        hidden_size = int(mod.get('hidden_size', 32))
        num_layers = int(mod.get('num_layers', 2))
        print('Layer size: %d' % hidden_size)
        print('Number of layers: %d' % num_layers)
        # The dense policy network should have two hidden layers, each with
        # <obs dim> units (thereabouts, anyway)
        custom_network = make_masked_mlp('simple_masked_mlp', obs_dim, act_dim,
                                         (hidden_size, ) * num_layers)
    elif args.model == 'actprop':
        known_mo = {'hidden_size', 'num_layers', 'dropout', 'norm_response'}
        mod = args.model_opts
        hs = int(mod.get('hidden_size', 8))
        num_layers = int(mod.get('num_layers', 2))
        dropout = float(mod.get('dropout', 0.0))
        norm_response = int(mod.get('norm_response', '0')) != 0
        print('hidden_size: %d, num_layers: %d, dropout: %f, norm_response: %d'
              % (hs, num_layers, dropout, int(norm_response)))
        if weight_manager is not None:
            print('Re-using same weight manager')
        elif args.resume_from:
            print('Reloading weight manager (resuming training)')
            weight_manager = joblib.load(args.resume_from)
        else:
            print('Creating new weight manager (not resuming)')
            # TODO: should save all network metadata, including heuristic
            # configuration
            # extra_dim = sum([g.extra_dim for g in data_gens])
            weight_manager = PropNetworkWeights(
                dom_meta,
                hidden_sizes=[(hs, hs)] * num_layers,
                # extra inputs to each action module from data generators
                extra_dim=dg_extra_dim)
        custom_network = PropNetwork(
            weight_manager,
            prob_meta,
            dropout=dropout,
            norm_response=norm_response)
    else:
        raise ValueError('Unknown network type "%s"' % args.model)

    # What if a model option wasn't used?
    unknown_mo = args.model_opts.keys() - known_mo
    if unknown_mo:
        print(
            'WARNING: model options not understood by "%s" network: %s' %
            (args.model, ', '.join(unknown_mo)),
            file=sys.stderr)

    policy = CategoricalMLPPolicy(
        env_spec=env_spec, prob_network=custom_network, name='policy')

    # weight_manager will sometimes be None
    return policy, weight_manager


def get_problem_names(pddl_files):
    """Return a list of problem names from some PDDL files by spooling up
    background process."""
    config = ProblemServiceConfig(pddl_files, None)
    server = ProblemServer(config)
    try:
        names = to_local(server.service.get_problem_names())
        assert isinstance(names, list)
        assert all(isinstance(name, str) for name in names)
    finally:
        server.stop()
    return names


class SingleProblem(object):
    """Wrapper to store all information relevant to training on a single
    problem."""

    def __init__(self, name, problem_server):
        self.name = name
        # need a handle to problem server so that it doesn't get GC'd (which
        # would kill the child process!)
        self.problem_server = problem_server
        self.problem_service = problem_server.service
        self.prob_meta, self.dom_meta = to_local(
            self.problem_service.get_meta())
        self.env_spec = to_local(self.problem_service.get_env_spec())
        self.dg_extra_dim = to_local(self.problem_service.get_dg_extra_dim())
        # will get filled in later
        self.policy = None


def make_services(args):
    """Make a ProblemService for each relevant problem."""
    # first get names
    if not args.problems:
        print("No problem name given, will use all discovered problems")
        problem_names = get_problem_names(args.pddls)
    else:
        problem_names = args.problems
    print("Loading problems %s" % ', '.join(problem_names))

    # now get contexts for each problem and a manager for their weights
    servers = []

    def kill_servers():
        for server in servers:
            try:
                server.stop()
            except Exception as e:
                print("Got exception %r while trying to stop %r" % (e, server))

    atexit.register(kill_servers)

    for problem_name in problem_names:
        service_config = ProblemServiceConfig(
            args.pddls,
            problem_name,
            heuristic_name=args.heuristic,
            teacher_heur=args.supervised_teacher_heur,
            use_lm_cuts=args.use_lm_cuts,
            dump_table_path=args.dump_table_path,
            dump_table_interval=args.dump_table_interval)
        problem_server = ProblemServer(service_config)
        servers.append(problem_server)

    # do this as a separate loop so that we can wait for services to spool
    # up in background
    # TODO: make the on_connect methods for each ProblemService run
    # asynchronously, too! That should let SSiPP and MDPSim ground in
    # parallel.
    problems = []
    weight_manager = None
    for problem_name, problem_server in zip(problem_names, servers):
        print('Starting service and policy for %s' % problem_name)
        problem = SingleProblem(problem_name, problem_server)
        problem.policy, weight_manager = make_policy(
            args,
            problem.env_spec,
            problem.dom_meta,
            problem.prob_meta,
            problem.dg_extra_dim,
            weight_manager=weight_manager)
        problems.append(problem)

    return problems, weight_manager


def main_supervised(args, unique_prefix, snapshot_dir, scratch_dir):
    print('Training supervised')

    start_time = time()
    problems, weight_manager = make_services(args)

    # need to create FileWriter and give it to CategoricalMLPPolicy *after*
    # creating the policy network itself, or the network will not show up in TB
    summary_path = path.join(scratch_dir, 'tensorboard')
    sample_writer = tf.summary.FileWriter(
        summary_path, graph=tf.get_default_graph())
    for problem in problems:
        problem.policy.summary_writer = sample_writer

    with ExitStack() as stack:
        sess = tf.Session()
        stack.enter_context(sess)
        # need to keep session around for as long as we want to use the
        # network which we build
        if args.debug:
            print('Enabling TensorFlow debugger')
            sess = tfdbg.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
            stack.enter_context(sess.as_default())

        if not args.no_train:
            print('Training supervised')
            sup_trainer = SupervisedTrainer(
                problems=problems,
                weight_manager=weight_manager,
                summary_writer=sample_writer,
                strategy=SupervisedObjective.ANY_GOOD_ACTION,
                # strategy=SupervisedObjective.MAX_ADVANTAGE,
                kl_coeff=args.supervised_kl_coeff,
                batch_size=args.supervised_bs,
                lr=args.supervised_lr)
            train_gen = sup_trainer.train()
            best_rate = None
            iter_num = 0
            time_since_best = 0
            succ_rate, mean_loss, replay_size = next(train_gen)
            keep_going = True
            while True:
                elapsed_time = time() - start_time
                # This timeout mechanism is not very accurate, but it doesn't
                # matter, since the runner will terminate us after we timeout
                # anyway.
                if args.timeout:
                    keep_going = keep_going and elapsed_time <= args.timeout
                try:
                    succ_rate, mean_loss, replay_size = train_gen.send(
                        keep_going)
                except StopIteration:
                    break
                iter_num += 1

                if best_rate is None or succ_rate > best_rate + 1e-4:
                    time_since_best = 0
                else:
                    time_since_best += 1
                    if time_since_best >= args.supervised_early_stop \
                            and succ_rate >= 0.999:
                        print('Terminating (early stopping condition met '
                              'with %d epochs since loss %f)' %
                              (time_since_best, best_rate))
                        keep_going = False

                if best_rate is None or succ_rate >= best_rate:
                    best_rate = succ_rate
                    # snapshot!
                    snapshot_path = path.join(snapshot_dir,
                                              'snapshot_%d_%f.pkl' %
                                              (iter_num, succ_rate))
                    weight_manager.save(snapshot_path)
        else:
            # need to fill up stats values with garbage :P
            elapsed_time = iter_num = None
            # normally trainers do this
            sess.run(tf.global_variables_initializer())

        # evaluate
        if weight_manager is not None:
            weight_manager.save(path.join(snapshot_dir, 'snapshot_final.pkl'))
        for problem in problems:
            print('Evaluating on %s' % problem.name)
            eval_single(args, problem.policy, problem.problem_server,
                        unique_prefix + '-' + problem.name, elapsed_time,
                        iter_num, weight_manager, scratch_dir)


def main():
    # this is required for rpyc to allow pickling
    rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

    args = parser.parse_args()

    unique_prefix = unique_name(args)
    print('Unique prefix:', unique_prefix)

    if args.expt_dir is None:
        args.expt_dir = 'experiment-results'
    scratch_dir = path.join(args.expt_dir, unique_prefix)
    makedirs(scratch_dir, exist_ok=True)

    # log directory
    log_file = path.join(scratch_dir, 'logs.csv')
    print('Log file:', log_file)
    logger.add_tabular_output(log_file)

    # where to save models
    snapshot_dir = path.join(scratch_dir, 'snapshots')
    makedirs(snapshot_dir, exist_ok=True)
    print('Snapshot directory:', snapshot_dir)
    # I *won't* use next line (setting up rllab snapshot handling) because I
    # want to handle snapshots myself
    # logger.set_snapshot_dir(snapshot_dir)

    if args.supervised:
        main_supervised(args, unique_prefix, snapshot_dir, scratch_dir)
    else:
        main_rllab(args, unique_prefix, snapshot_dir)


if __name__ == '__main__':
    # these will be useful for nefarious hacking when running under kernprof
    import prof_utils
    prof_utils._fpg_main_globals = globals()

    # now run actual program
    main()
