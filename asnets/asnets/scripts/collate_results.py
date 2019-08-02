#!/usr/bin/env python3
"""Collate results for ASNets, from `run_experiment.py`."""

import numpy as np

import argparse
from ast import literal_eval
from collections import namedtuple
from json import load, dumps
import os
import re
import sys


def parse_name_with_nums(name):
    non_num_parts = re.split(r'\d+', name)
    pointer = 0
    rv = []
    for non_num_part in non_num_parts:
        rv.append(non_num_part)
        pointer += len(non_num_part)
        if pointer >= len(name):
            continue
        num_match = re.findall('^\d+', name[pointer:])[0]
        num_comp = int(num_match)
        pointer += len(num_match)
        rv.append(num_comp)
    return tuple(rv)


def name_to_size(name):
    parsed = parse_name_with_nums(name)
    for item in parsed:
        if isinstance(item, int):
            # choose first number in name as prob size
            return item
    raise ValueError("Couldn't get size for problem '%s'" % name)


def print_expt_results(full_results):
    print('Experiments with architecture <strong>{arch_name}</strong> on '
          'environment <strong>{env_name}</strong>. Trained for '
          '{train_time}s on files {train_probs}. {term_stat} (on training '
          'run).\n'.format(
              arch_name=full_results.arch_name,
              env_name=full_results.expt_name,
              train_time=full_results.train_time,
              train_probs=', '.join(full_results.train_problems),
              term_stat=full_results.train_exec_status))

    print('<table>')
    print(' <tr><th>Problem</th><th>Success rate</th>'
          '<th>Mean successful trial length</th><th>Evaluation time</th>'
          '<th>Termination status</th></tr>')

    sorted_runs = sorted(
        full_results.eval_runs, key=lambda r: parse_name_with_nums(r.name))
    for prob_result in sorted_runs:
        # entire result goes on one row in output
        print(' <tr>', end='')
        # problem name
        print('<td>%s</td>' % prob_result.name, end='')
        # success rate
        print(
            '<td>%d/%d (%.2f%%)</td>' %
            (np.sum(prob_result.goal_reached), prob_result.runs,
             100 * np.mean(prob_result.goal_reached)),
            end='')
        # mean successful trial length
        print(
            '<td>%.2f (limited to %d)</td>' %
            (np.mean(prob_result.costs), prob_result.turn_limit),
            end='')
        # evaluation time
        print(
            '<td>%.2fs (%.2fs/run)</td>' %
            (np.sum(prob_result.eval_times), np.mean(prob_result.eval_times)),
            end='')
        # termination status
        print('<td>%s</td>' % prob_result.eval_exec_status, end='')
        # end row
        print('</tr>')

    print('</table>')


def print_expt_json(full_results, dest=sys.stdout):
    """Dump results as JSON for processing by plotting script"""
    sorted_runs = sorted(
        full_results.eval_runs, key=lambda r: parse_name_with_nums(r.name))
    prob_sizes = []
    prob_names = []
    eval_runs = []
    for prob_result in sorted_runs:
        name = prob_result.name
        size = name_to_size(name)
        eval_run = {
            'goal_reached': prob_result.goal_reached,
            'time': prob_result.eval_times,
            'cost': prob_result.costs,
        }
        eval_runs.append(eval_run)
        prob_sizes.append(size)
        prob_names.append(name)

    result = {
        'eval_names': prob_names,
        'eval_sizes': prob_sizes,
        'eval_runs': eval_runs,
        'name_arch': full_results.arch_name,
        'name_expt': full_results.expt_name,
        'train_time': full_results.train_time,
    }

    print(dumps(result, indent=2, sort_keys=True), file=dest)


def get_run_info(run_dir):
    """Parse relevant fines from run-info subdirectory attached to each
    experiment."""
    term_stat_path = os.path.join(run_dir, 'termination_status')
    with open(term_stat_path, 'r') as fp:
        # usual keys that end up in term_stat are timed_out and bad_retcode
        term_stat = {}
        for line in fp:
            if not line:
                continue
            head, tail = line.split(':', 1)
            head = head.strip()
            tail = tail.strip()
            term_stat[head] = literal_eval(tail)

    elapsed_time_path = os.path.join(run_dir, 'elapsed_secs')
    with open(elapsed_time_path, 'r') as fp:
        elapsed_time_str = fp.read()
        elapsed_time = float(elapsed_time_str)

    cmdline_path = os.path.join(run_dir, 'cmdline')
    with open(cmdline_path, 'r') as fp:
        cmdline = fp.read().strip()

    is_train = os.path.exists(os.path.join(run_dir, 'is_train'))

    return term_stat, elapsed_time, cmdline, is_train


def format_term_stat(term_stat):
    """Turn term_stat dict from get_run_info into something human-readable."""
    if term_stat['timed_out']:
        desc = 'Killed by timeout; '
    else:
        desc = 'Ran to completion; '
    if term_stat['bad_retcode']:
        desc += 'signalled error on termination'
        if term_stat['timed_out']:
            desc += ", but probably just because it was SIGKILL'd after " \
                    "timeout"
    else:
        desc += 'did not signal error'
    return desc


def problems_from_cmdline(cmdline):
    # try to extract a list of PDDL file basenames given on command line
    flags = cmdline.strip().split()[1:]
    probs = []
    for f in flags:
        if f.startswith('-'):
            continue
        exts = ['.pddl', '.ppddl']
        for ext in exts:
            if f.endswith(ext):
                bn = os.path.basename(f)
                probs.append(bn[:-len(ext)])
                break
    if not probs:
        probs = ['(unknown problems?)']
    return probs


ProblemResult = namedtuple('ProblemResult', [
    'name', 'runs', 'goal_reached', 'eval_times', 'costs', 'eval_exec_status',
    'turn_limit'
])
FullRunResult = namedtuple('FullRunResult', [
    'arch_name', 'expt_name', 'train_problems', 'train_time',
    'train_exec_status', 'eval_runs'
])


def parse_results_dir(results_dir):
    # should have a bunch of subdirectories of the form
    # P[name1,name2,...]-O[...]-...
    print('Processing results in %s' % results_dir)

    results_dir_strip = results_dir.rstrip(os.path.sep)
    results_dir_bn = os.path.basename(results_dir_strip)
    expt_mod_name, arch_mod_name = results_dir_bn.split('-')[:2]
    expt_name = expt_mod_name.split('.')[-1]
    arch_name = arch_mod_name.split('.')[-1]
    print('Experiment name: %s' % expt_name)
    print('Architecture name: %s' % arch_name)

    subdirs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        # skip "results" subdir and whatever other cruft got stuck in here
        if d.startswith('P[')
    ]
    eval_runs = []
    train_time = 'unknown train time'
    # TODO: figure out how I can do this :/
    train_problems = ['?unknown problems?']
    train_exec_status = 'Unknown termination status'
    seen_train_dir = False
    for subdir in subdirs:
        results_path = os.path.join(subdir, 'results.json')

        try:
            run_dir = os.path.join(subdir, 'run-info')
            term_stat, elapsed_time, cmdline, is_train = get_run_info(run_dir)
        except FileNotFoundError:
            print("Can't even find run info for '%s'; skipping" % subdir)
            continue

        try:
            with open(results_path, 'r') as fp:
                results_dict = load(fp)
        except FileNotFoundError as e:
            print("Couldn't find results.json in '%s'; skipping" % subdir)
            if not is_train:
                continue

        # things of interest: no_train, problem, trials, successes,
        # mean_makespan, turn_limit
        if is_train:
            if seen_train_dir:
                # we can't have two train dirs because we can't merge stats for
                # train time, execution status, etc. (sometimes this issue
                # happens when I try to manually extend the training period for
                # a problem, instead of just using run_experiment.py like a
                # normal human being)
                raise Exception(
                    "Two train dirs (%s and %s)---maybe delete one?" %
                    (seen_train_dir[1], subdir))
            print("Skipping %s, as it's for a training run" % subdir)
            seen_train_dir = (True, subdir)
            # this is a training run
            train_time = elapsed_time
            train_exec_status = format_term_stat(term_stat)
            train_problems = problems_from_cmdline(cmdline)
            continue

        eval_run = ProblemResult(
            name=results_dict['problem'],
            runs=int(results_dict['trials']),
            goal_reached=results_dict['all_goal_reached'],
            eval_times=results_dict['all_exec_times'],
            costs=results_dict['all_costs'],
            eval_exec_status=format_term_stat(term_stat),
            turn_limit=int(results_dict['turn_limit']))
        eval_runs.append(eval_run)

    full_results = FullRunResult(
        arch_name=arch_name,
        expt_name=expt_name,
        train_problems=train_problems,
        train_time=train_time,
        train_exec_status=train_exec_status,
        eval_runs=eval_runs)

    return full_results


parser = argparse.ArgumentParser(
    description='Format results from run_experiment.py')
parser.add_argument(
    '--json-dir',
    default=None,
    help='write .json files to this directory as well as stdout')
parser.add_argument(
    'results_dir',
    metavar='results-dir',
    nargs='+',
    help='path to directories holding results')


def main():
    args = parser.parse_args()
    for results_dir in args.results_dir:
        results = parse_results_dir(results_dir)
        if not args.json_dir:
            print('')
            print('== HTML RESULTS')
            print('-' * 80)
            print_expt_results(results)
            print('-' * 80)
            print('')
        else:
            os.makedirs(args.json_dir, exist_ok=True)
            # should make suffix unique
            results_dir = os.path.basename(results_dir.rstrip(os.sep))
            out_fn = 'results-%s.json' % results_dir
            json_out_path = os.path.join(args.json_dir, out_fn)
            print('Writing JSON to "%s"' % json_out_path)
            with open(json_out_path, 'w') as fp:
                print_expt_json(results, dest=fp)


if __name__ == '__main__':
    main()
