#!/usr/bin/env python3
"""Run probabilistic baselines with SSiPP."""

import collections
import datetime
import functools
import hashlib
import itertools
import math
import json
import os
import random
import re
import struct
import subprocess
import time

import click
import ray

from asnets.scripts.run_det_baselines import get_module_test_problems, \
    wait_all_unordered, ray_setup, ASNETS_ROOT
from asnets.pddl_utils import extract_domain_problem, extract_domain_name
from asnets.ssipp_interface import get_ssipp_solver_path_auto

# each combination of planner & heuristic
PLANNERS = list(itertools.product(['lrtdp', 'ssipp'], ['h-add', 'lm-cut']))
PLANNER_IDENTS = {
    # shorthand for complete SSiPP planners
    'lrtdp': 'lrtdp',
    'ssipp': 'ssipp:lrtdp:max_depth:3'
}

# cheesy name :)
OrDict = collections.OrderedDict


class PlannerError(Exception):
    """Get raised when SSiPP does something we don't expect (e.g returns some
    output that we couldn't parse)."""
    pass


def make_int_seed(shit_to_hash):
    """Make random 31-bit seed out of any Python data structure that can be
    converted to a string :) (this is sooooo brittle)"""
    bytes_shit = str(shit_to_hash).encode('utf8')
    md5 = hashlib.md5(bytes_shit)
    big_num, = struct.unpack('>I', md5.digest()[:4])
    # limit to 31 bits
    return big_num & 0x3fffffff


def problem_seed_iterator(prob_path, prob_name, repeats):
    base_seed = make_int_seed([prob_path, prob_name])
    seeds = set()
    for launch_num in range(repeats):
        new_seed = str(base_seed ^ make_int_seed(launch_num))
        assert new_seed not in seeds
        seeds.add(new_seed)
        yield new_seed


def process_job(output, rcode):
    avg_cost_match = re.search(r'^Observed Avg cost = (\d\.\d+|\d+)$', output,
                               re.MULTILINE)
    cpu_time_match = re.search(r'^CPU\+Sys time: (\d+)$', output, re.MULTILINE)
    wall_time_match = re.search(r'^Wall time: (\d+)$', output, re.MULTILINE)
    status_match = re.search(
        r'^\[Round Summary\]: status = '
        r'(deadend-reached|goal-reached|max-turns-reached)$', output,
        re.MULTILINE)
    nstates_match = re.search("^\[state-hash\]: number states = (\d+)$",
                              output, re.MULTILINE)
    if not all((avg_cost_match, cpu_time_match, wall_time_match, status_match,
                nstates_match)):
        return None
    # example result dict from Felipe's SSiPP runs:
    # {"goal_reached": 1, "observed_cost": 40, "visited_states": 40,
    #  "wall_time": 2328083043, "cpu_time": 2329440000}
    observed_cost = int(avg_cost_match.groups()[0])
    cpu_time = int(cpu_time_match.groups()[0])
    wall_time = int(wall_time_match.groups()[0])
    visited_states = int(nstates_match.groups()[0])
    goal_reached = int(status_match.groups()[0] == 'goal-reached')
    return OrDict([
        ("goal_reached", goal_reached),
        ("observed_cost", observed_cost),
        ("visited_states", visited_states),
        ("wall_time", wall_time),
        ("cpu_time", cpu_time),
    ])


def ssipp_command(*, ssipp_path, domain_path, prob_path, prob_name, planner,
                  heuristic, max_secs, seed):
    """Prepare SSiPP command line appropriately for given options."""
    command = [
        # executable
        ssipp_path,
        # common options (incl 16GB memory limit)
        *'-R 1 --max_rss_kb 16777216 -d 1000 -M 1000'.split(),
        # time limit handling is somewhat difficult because trained SSiPP
        # makes things a pain
        '--max_cpu_time_sec',
        str(max_secs),
        # heuristic spec '-h',
        '-h',
        heuristic,
        # random seed
        '-r',
        seed,
        '-p',
        PLANNER_IDENTS[planner],
    ]

    if planner == 'ssipp':
        # this is "trained SSiPP", so we need an extra flag to handle
        # training duration
        command.extend(['--train_for', str(max(1, max_secs - 10))])

    command.extend([domain_path, prob_path, prob_name])

    return command


def do_problem_trials(*, secs_per_run, secs_total, repeats, **ssipp_kwargs):
    time_elapsed = 0.0
    # maps seeds to results for the seed
    return_dict = OrDict()

    seed_iterator = problem_seed_iterator(ssipp_kwargs['prob_path'],
                                          str(ssipp_kwargs['prob_name']),
                                          repeats)
    for seed in seed_iterator:
        run_budget = min(secs_per_run, secs_total - time_elapsed)
        if run_budget < 1:
            # we've timed out, so fill up this slot with None (downstream tools
            # know how to deal with that)
            return_dict[seed] = None
            continue
        # run budget needs to be integer-valued and nonzero (internally it gets
        # converted with atoi, so fractional part is ignored; further, a limit
        # of 0 is interpreted as "run forever")
        run_budget = int(math.ceil(run_budget))
        command = ssipp_command(max_secs=run_budget, seed=seed, **ssipp_kwargs)
        command_start = time.time()
        command_result = subprocess.run(command,
                                        check=False,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
        time_elapsed += time.time() - command_start
        # command_result is a CompletedProcess that has {args, returncode,
        # stdout, stderr}
        out_dict = process_job(command_result.stdout,
                               command_result.returncode)
        if command_result.returncode == 0 and out_dict is None:
            # SHOUT ABOUT THIS
            raise PlannerError(
                "Return code of command %s is 0, but could not parse output."
                "Here is stdout:\n%s\n\nHere is stderr:\n%s\n\n"
                "Is this a bug?" % (command_result.args, command_result.stdout,
                                    command_result.stderr))
        return_dict[seed] = out_dict

    assert len(return_dict) == repeats

    return return_dict


def make_run_ident(planner, heuristic, secs_total):
    return '%s:%s:t%ss' % (PLANNER_IDENTS[planner], heuristic, secs_total)


def sort_dict(some_dict):
    """Convert a dictionary into an OrderedDict with keys sorted
    using Python's sort order."""
    return OrDict(sorted(some_dict.items(), key=lambda kv: kv[0]))


@ray.remote(num_cpus=1)
def do_experiment(planner, heuristic, domain_path, problem_path, problem_name,
                  ssipp_path, secs_per_run, secs_total, repeats):
    os.chdir(ASNETS_ROOT)

    ssipp_kwargs = dict(
        ssipp_path=ssipp_path,
        domain_path=domain_path,
        prob_path=problem_path,
        prob_name=problem_name,
        planner=planner,
        heuristic=heuristic,
    )

    result_dict = do_problem_trials(repeats=repeats,
                                    secs_total=secs_total,
                                    secs_per_run=secs_per_run,
                                    **ssipp_kwargs)
    # did ANY of the runs succeed? (if not, we may want to start killing later
    # runs)
    any_success = any(d is not None for d in result_dict.values())
    run_ident = make_run_ident(planner, heuristic, secs_per_run)
    # final dictionary is keyed by run_ident so we can put it straight into
    # output dict structure that gets serialised
    final_dict = OrDict([(run_ident, result_dict)])

    # job & success must be returned as first two args; last arg is
    # whatever you want it to be (executor_loop is where this API is used)
    return any_success, problem_name, final_dict


@click.command(help='run probabilistic baseline planners using Ray')
@click.argument('problem_module_name')
@click.option('--ssipp-path',
              default=None,
              type=str,
              help='path to SSiPP solver_ssp')
@click.option('--repeat',
              '-r',
              default=30,
              help='number of times to repeat experiment')
@click.option(
    '--out-dir',
    '-o',
    # I put things in ./experiment-results/ so that they get stored
    # to NFS when I run on AWS
    default='./experiment-results/baselines-prob/',
    help='output directory to put results in')
@click.option(
    '--max-sec-per-run',
    default=3 * 60 * 60,  # 3h (900m)
    help='maximum duration of a single planner execution (in seconds)')
@click.option(
    '--max-sec-per-problem',
    default=3 * 60 * 60,  # 3h (900m; this is a LOT of time)
    help='maximum duration of all executions on a problem (in seconds)')
@click.option(
    '--ray-connect',
    default=None,
    type=str,
    help='connect Ray to this Redis DB instead of starting new cluster')
@click.option(
    '--ray-ncpus',
    default=None,
    type=int,
    help='restrict Ray pool to use this many CPUs *in total* (only valid if '
    'spinning up new Ray cluster)')
def main(problem_module_name, ssipp_path, repeat, out_dir, max_sec_per_run,
         max_sec_per_problem, ray_connect, ray_ncpus):
    # timing sanity check; can't complete a problem in less time than it takes
    # for a single run
    assert 0 < max_sec_per_run <= max_sec_per_problem, \
        "must have 0 < --max-sec-per-run <= --max-sec-per-problem"

    # always run from ASNETS_ROOT so that references to ./experiment-results/
    # go through correctly (I have NFS volume symlinked to there on AWS)
    os.chdir(ASNETS_ROOT)

    # find SSiPP installation
    ssipp_path = ssipp_path or get_ssipp_solver_path_auto()
    print("Assuming SSiPP at %s" % (ssipp_path, ))

    # spin up Ray
    ray_setup(ray_connect, ray_ncpus)

    # get problem location info
    print('Importing problem from %s' % problem_module_name)
    domain_path, test_prob_paths_raw = get_module_test_problems(
        problem_module_name)
    domain_name = extract_domain_name(domain_path)
    all_problems = []
    for test_prob_path, prob_name in test_prob_paths_raw:
        if prob_name is None:
            _, _, _, prob_name = extract_domain_problem(
                [domain_path, test_prob_path])
        all_problems.append((test_prob_path, prob_name))

    # run!
    print("Launching tasks")
    task_partials = []
    for problem_path, problem_name in all_problems:
        for planner, heuristic in PLANNERS:
            partial = functools.partial(do_experiment.remote, planner,
                                        heuristic, domain_path, problem_path,
                                        problem_name, ssipp_path,
                                        max_sec_per_run, max_sec_per_problem,
                                        repeat)
            task_partials.append(partial)
    random.shuffle(task_partials)
    tasks = [part() for part in task_partials]
    finished_jobs = wait_all_unordered(tasks)

    # now we collect all results into an output dict keyed by problem name,
    # where values are themselves dictionaries keyed by run IDs
    out_dict = {}
    for any_success, problem_name, singleton_dict in finished_jobs:
        # singleton_dict is keyed by run ID & maps to a dict of form {seed1:
        # run_info1, seed2: run_info2, …}
        out_dict.setdefault(problem_name, {}).update(singleton_dict)

    # order the output dict by name, and also order the second-level dict by
    # run ident
    out_dict = sort_dict({k: sort_dict(v) for k, v in out_dict.items()})

    # finally write results JSON file
    date_str = datetime.datetime.now().isoformat()
    out_fn = 'results-%s-%ds-%s.json' \
        % (domain_name, max_sec_per_problem, date_str)
    out_path = os.path.join(out_dir, out_fn)
    os.makedirs(out_dir, exist_ok=True)
    print("Writing output to '%s'" % (out_path, ))
    with open(out_path, 'w') as out_fp:
        json.dump(out_dict, out_fp)


if __name__ == '__main__':
    main()
