#!/usr/bin/env python3
"""Splits out baseline stats from Felipe's JSON file (similar to what
collate_results.py does for ASNet stats)."""

import argparse
from itertools import groupby
from json import load, dump
import os

from asnets.scripts.collate_results import (parse_name_with_nums, name_to_size)


def get_domains_problems(baseline_dict):
    """Turns a list of problem names like {'triangle-tire-1': ...,
    'cosanostra-n5': ..., 'triangle-tire-3': ..., ...} into a list of [(domain
    name, [problem names])] pairs."""
    result_keys = baseline_dict.keys()
    parsed = sorted(((parse_name_with_nums(n), n) for n in result_keys))

    def group_key(thing):
        strs = tuple(t for t in thing[0] if isinstance(t, (str, bytes)))
        if strs[0] in {'prob_bw_', 'ex_bw_'}:
            # HACK to handle exbw/prob bw naming (they've got weird suffixes,
            # but we still want to group them together)
            strs = strs[:1]
        return (len(thing), ) + strs

    rv = []
    for _, group_and_parsed in groupby(parsed, key=group_key):
        # group is list, each item in group is (parsed name, name) pair
        group = list(group_and_parsed)
        prob_names = [g[1] for g in group]
        dom_name = '-'.join(
            s.strip('-') for s in group[0][0] if isinstance(s, str) and s)
        rv.append((dom_name, prob_names))
    return rv


def get_planner_names(baseline_dict):
    """Like get_domains_problems, turn a dictionary read from a JSON file of
    baseline results into a list of names of baseline planners."""
    rv = set()
    for subdict in baseline_dict.values():
        rv.update(subdict.keys())
    return rv


def join_results(baseline_dict, dom_name, prob_names, planner_name):
    # should have a bunch of subdirectories of the form
    # P[name1,name2,...]-O[...]-...
    prob_sizes = []
    eval_runs = []
    for prob_name in prob_names:
        run_result = {}
        prob_results = baseline_dict[prob_name]
        if planner_name not in prob_results:
            # it probably timed out
            continue
        prob_planner_res = prob_results[planner_name]

        wall_times = []
        tot_costs = []
        reach_flags = []
        for run_dict in prob_planner_res.values():
            if run_dict is None:
                # this run timed out
                # we'll set its cost and wall time to None
                # things that don't reach the goal shouldn't count toward mean
                # cost or execution time (I don't think)
                wall_times.append(None)
                tot_costs.append(None)
                reach_flags.append(False)
            else:
                wall_times.append(run_dict['wall_time'] / 1.0e6)
                tot_costs.append(run_dict['observed_cost'])
                reach_flags.append(run_dict['goal_reached'] != 0)

        prob_size = name_to_size(prob_name)
        prob_sizes.append(prob_size)

        run_result = {
            'goal_reached': reach_flags,
            'time': wall_times,
            'cost': tot_costs,
        }
        eval_runs.append(run_result)

    result = {
        'eval_names': prob_names,
        'eval_sizes': prob_sizes,
        'eval_runs': eval_runs,
        'planner_name': planner_name
    }

    return result


parser = argparse.ArgumentParser(
    description='Format results from AAAI report baselines')
parser.add_argument(
    '--out-dir', metavar='out-dir', help='write .json files to this directory')
parser.add_argument(
    'input_jsons',
    metavar='input-jsons',
    nargs='+',
    help='path to .json files with baseline results; will be merged if '
    'necessary, with later files taking precedence')


def main():
    args = parser.parse_args()
    baseline_dict = {}
    for json_path in args.input_jsons:
        with open(json_path, 'r') as fp:
            new_dict = load(fp)
            for k, v in new_dict.items():
                if k in baseline_dict:
                    print('Merging results for "%s"' % k)
                    baseline_dict[k].update(v)
                else:
                    baseline_dict[k] = v
    domains_probs = get_domains_problems(baseline_dict)
    planner_names = get_planner_names(baseline_dict)
    for domain, prob_set in domains_probs:
        for planner_name in planner_names:
            results = join_results(baseline_dict, domain, prob_set,
                                   planner_name)
            json_bn = '%s-%s.json' % (domain, planner_name)
            json_path = os.path.join(args.out_dir, json_bn)
            print('Writing to "%s"' % json_path)
            os.makedirs(args.out_dir, exist_ok=True)
            with open(json_path, 'w') as fp:
                dump(results, fp)


if __name__ == '__main__':
    main()
