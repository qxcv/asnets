#!/usr/bin/env python3
"""Hyperparameter tuning with Ray Tune and Hyperopt. Inner loop does
combination of run_experiment and collate_results. Overall optimisation
objective is coverage as a fraction of the total number of problems."""

from argparse import ArgumentParser
from collections import OrderedDict
import copy
from importlib import import_module
import os
import multiprocessing

import numpy as np
import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from skopt.optimizer import Optimizer

from asnets.scripts.collate_results import parse_results_dir
from asnets.scripts.run_experiment import main_inner


class ModuleCopy(object):
    """For deep-copying attributes of modules (modules cannot be deepcopy'd
    directly)."""

    def __init__(self, module):
        whitelist = {'__name__'}  # special attrs that we CAN copy
        for attr_name in dir(module):
            if attr_name.startswith('_') and attr_name not in whitelist:
                continue
            orig_value = getattr(module, attr_name)
            copied = copy.deepcopy(orig_value)
            setattr(self, attr_name, copied)


# @ray.remote(num_cpus=0)
def inner_perform_trial(configured_arch_mod, base_prob_mod):
    """Perform trial on just one problem & return coverage"""
    prefix_dir = main_inner(
        arch_mod=configured_arch_mod,
        prob_mod=base_prob_mod,
        job_ncpus=1,
        enforce_job_ncpus=True)
    results = parse_results_dir(prefix_dir)
    # report coverage as a number between 0 & 1 indicating mean number of
    # trials that reached the goal
    coverage = np.sum([
        np.mean(run.goal_reached) if run.goal_reached else 0.0
        for run in results.eval_runs
    ])
    max_coverage = len(base_prob_mod.TEST_RUNS)
    norm_coverage = coverage / max(max_coverage, 1)
    return norm_coverage


def make_perform_trial(base_arch_mod, base_prob_mods):
    """Make function that performs a single trial for Ray Tune. Useful b/c it
    allows you to pass in extra info like the contents of the base module, the
    contents of the architecture module, etc."""
    above_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

    def perform_trial(config, reporter):
        """Perform trials on a group of problems & return sum of coverages."""
        os.chdir(above_dir)  # fuck Ray's bullshit
        configured_arch_mod = ModuleCopy(base_arch_mod)
        for k, v in config.items():
            assert hasattr(configured_arch_mod, k.upper()), \
                f'module missing attr {k.upper()}'
            setattr(configured_arch_mod, k.upper(), v)
        # XXX: .remote() calls broke on my Ray cluster, possibly because Tune
        # runs this in a thread, and calling ray.get() from a thread is meant
        # to be a bad idea?
        # coverages = ray.get([
        #     inner_perform_trial.remote(configured_arch_mod, prob_mod)
        #     for prob_mod in base_prob_mods
        # ])
        coverages = [
            inner_perform_trial(configured_arch_mod, prob_mod)
            for prob_mod in base_prob_mods
        ]
        covs_by_name = {
            pm.__name__: cov
            for pm, cov in zip(base_prob_mods, coverages)
        }
        sum_coverage = sum(coverages)
        reporter(coverage=sum_coverage, coverage_by_name=covs_by_name)

    return perform_trial


def main(args):
    # 1. load config
    print('Importing architecture from %s' % args.arch_module)
    arch_mod = import_module(args.arch_module)
    prob_mods = []
    for prob_module_path in args.prob_modules:
        print('Importing problem from %s' % prob_module_path)
        this_prob_mod = import_module(prob_module_path)
        prob_mods.append(this_prob_mod)

    # 2. spool up Ray
    new_cluster = args.ray_connect is None
    ray_kwargs = {}
    if not new_cluster:
        ray_kwargs["redis_address"] = args.ray_connect
        assert args.ray_ncpus is None, \
            "can't provide --ray-ncpus and --ray-connect"
    else:
        if args.ray_ncpus is not None:
            assert args.job_ncpus is None \
                    or args.job_ncpus <= args.ray_ncpus, \
                    "must have --job-ncpus <= --ray-ncpus if both given"
            ray_kwargs["num_cpus"] = args.ray_ncpus
    ray.init(**ray_kwargs)

    max_par_trials = args.max_par_trials
    if max_par_trials is None:
        # leave some room for hyperthread-caused over-counting of CPUs (a /2
        # factor), and for running eval trials in parallel
        max_par_trials = max(1, multiprocessing.cpu_count() // 5)

    sk_space = OrderedDict()
    # originally I had this split between 2/3, but I think 3 is a bit too slow
    # on some problems, so I want to stick to 2 (even though exbw really seems
    # to benefit from 3)
    sk_space['num_layers'] = [2]
    sk_space['hidden_size'] = (12, 20)
    # empty list; no steps down, just a single fixed learning rate
    sk_space['learning_rate_steps'] = [()]
    sk_space['supervised_learning_rate'] = (1e-4, 1e-2, 'log-uniform')
    # these ranges are similar to my original config, which seemed to work okay
    sk_space['supervised_batch_size'] = (48, 128)
    sk_space['opt_batch_per_epoch'] = (300, 1200)  # (150, 1500)
    # we use categorical vars to add "switched off entirely" as options (as
    # opposed to just "turned down very low"); I suspect switching off entirely
    # is good for some of those things
    sk_space['dropout'] = [0, 0.1, 0.25]
    sk_space['l1_reg'] = [0.0]  # (1e-10, 1e-2, 'log-uniform')
    sk_space['l2_reg'] = (1e-5, 1e-2, 'log-uniform')
    sk_space['target_rollouts_per_epoch'] = (30, 150)
    if arch_mod.TEACHER_PLANNER == 'ssipp':
        # only relevant for SSiPP
        # (originally I had both h-add and lm-cut as options, but lm-cut didn't
        # seem to help much, so I'm leaving it out)
        sk_space['ssipp_teacher_heuristic'] = ['h-add']

    # using random forest b/c we have lots of discrete params, & a few
    # categorical
    sk_optimiser = Optimizer(list(sk_space.values()), base_estimator='RF')
    algo = SkOptSearch(
        sk_optimiser,
        sk_space.keys(),
        max_concurrent=max_par_trials,
        metric='coverage',
        mode='max')

    perform_trial = make_perform_trial(arch_mod, prob_mods)
    tune.run(
        perform_trial,
        search_alg=algo,
        local_dir=args.work_dir,
        resources_per_trial={"cpu": 0},
        num_samples=1000)


parser = ArgumentParser()
parser.add_argument(
    '--work-dir', help='working directory', default='experiment-results/tune/')
parser.add_argument(
    '--ray-connect',
    default=None,
    help='connect Ray to this Redis DB instead of starting new cluster')
parser.add_argument(
    '--ray-ncpus',
    default=None,
    type=int,
    help='restrict Ray pool to use this many CPUs *in total* (only valid if '
    'spinning up new Ray cluster)')
parser.add_argument(
    '--max-par-trials',
    default=None,
    type=int,
    help='max number of trials to run in parallel')
parser.add_argument(
    'arch_module',
    metavar='arch-module',
    help='import path for Python file with architecture config (e.g. '
    '"experiments.actprop_1l")')
parser.add_argument(
    'prob_modules',
    metavar='prob-module',
    nargs='+',
    help='import paths for Python files with problem configs (e.g. '
    '"experiments.ex_blocksworld experiments.triangle_tireworld")')

if __name__ == '__main__':
    main(parser.parse_args())
