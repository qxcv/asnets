#!/usr/bin/env python3
"""Benchmark various planners on the IPC'08 learning track bootstrap & target
instances. Spits out HTML file. Is SUPER VULNERABLE to HTML injection (but it
shouldn't matter for my purposes)."""

import collections
import concurrent.futures as fu
import itertools as it
import os.path as osp
import os
import random
import time

import numpy as np
import tqdm

import asnets.fd_interface as fdi
import asnets.pddl_utils as pu

DOMAINS = ["gold-miner", "matching-bw", "n-puzzle", "parking", "sokoban"]
PLANNERS = [
    "astar-hadd", "astar-lmcount", "lama-w1", "lama-w2", "lama-w3",
    "lama-2011", "gbf-lmcut"
]
PROBLEM_ROOT = "../../../problems/ipc08-learn/"
# move this up or down depending on machine load
MAX_WORKERS = 8
REPEATS = 5
TIMEOUT = 5
OUT_FILE = 'planner-times.html'


def collect_instances(domain_name):
    # return list of (domain_name, instance_name, domain_path, instance_path)
    # tuples for a given domain
    prob_root = osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), PROBLEM_ROOT,
                 domain_name))
    learn_root = osp.join(prob_root, "learning")
    # use "learning/<domain>-typed.pddl" as domain and
    # "learning/{bootstrap,target}/typed/" as instances
    domain_path = osp.join(learn_root, "%s-typed.pddl" % domain_name)
    prob_dirs = [
        osp.join(learn_root, "bootstrap", "typed"),
        osp.join(learn_root, "target", "typed"),
    ]
    # in some cases I've added a directory with some extra train problems
    for possible_dir in [osp.join(prob_root, "mine", "train")]:
        if osp.exists(possible_dir):
            prob_dirs.append(possible_dir)
    rv_tuples = []
    for prob_dir in prob_dirs:
        for filename in sorted(os.listdir(prob_dir)):
            if not filename.endswith(".pddl"):
                continue
            instance_path = osp.join(prob_dir, filename)
            _, _, _, instance_name = pu.extract_domain_problem(
                [domain_path, instance_path])
            rv_tuples.append(
                (domain_name, instance_name, domain_path, instance_path))
    return rv_tuples


def execute(prob_tup):
    planner, domain_name, instance_name, domain_path, instance_path = prob_tup
    with open(domain_path, "r") as fp:
        domain_txt = fp.read()
    with open(instance_path, "r") as fp:
        problem_txt = fp.read()
    key = (domain_name, instance_name, planner)
    try:
        start = time.time()
        result = fdi.run_fd_or_timeout(planner,
                                       domain_txt,
                                       problem_txt,
                                       timeout_s=TIMEOUT)
        elapsed = time.time() - start
        plan_len = len(result)
    except fdi.FDTimeout:
        return key, False, None, None
    except Exception:
        print("Exception while working on %s" % (prob_tup, ))
        raise
    return key, True, elapsed, plan_len


def _mean_x(current_value, new_items, *, comp=min):
    if len(new_items) == 0:
        return current_value
    mu = np.mean(new_items)
    return comp([current_value, mu])


def mean_max(*args):
    return _mean_x(*args, comp=max)


def mean_min(*args):
    return _mean_x(*args, comp=min)


def main():
    print("Collecting instance names, etc.")
    all_instances = sum((collect_instances(dn) for dn in DOMAINS), [])
    instances_by_domain = {}
    for domain_name, instance_name, _, _ in all_instances:
        dom_dict = instances_by_domain.setdefault(domain_name,
                                                  collections.OrderedDict())
        # just put it in dict (we want to iterate over its keys in-order later
        # on, which is why we're not using a normal set)
        dom_dict[instance_name] = None
    all_tasks = list((p, ) + t for p, t in it.product(PLANNERS, all_instances))
    rep_tasks = list(it.chain.from_iterable(it.repeat(all_tasks, REPEATS)))
    random.shuffle(rep_tasks)

    print("%d tasks collected; running on pool with %d workers" %
          (len(rep_tasks), MAX_WORKERS))
    # these will be keyed on (domain, instance, planner)
    succs_dict = collections.defaultdict(lambda: [])
    timing_dict = collections.defaultdict(lambda: [])
    lens_dict = collections.defaultdict(lambda: [])
    with fu.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        exec_map = executor.map(execute, rep_tasks)
        map_wrapped = tqdm.tqdm(exec_map, total=len(rep_tasks))
        for key, succ, elapsed, length in map_wrapped:
            succs_dict[key].append(succ)
            if succ:
                timing_dict[key].append(elapsed)
                lens_dict[key].append(length)

    print("Building results table")
    html_segments = []
    for domain in DOMAINS:
        # prepare a new table for this domain
        table_rows = [[
            '<th>Instance</th>', *['<th>%s</th>' % p for p in PLANNERS]
        ]]
        for instance in instances_by_domain[domain]:
            # prepare a row for this instance
            row_contents = '<th>%s</th>' % instance
            best_succ = 0.0
            best_times = float('inf')
            best_lens = float('inf')
            row_it = []
            for planner in PLANNERS:  # columns correspond to PLANNERS
                # prepare a cell for this planner
                key = (domain, instance, planner)
                succs = succs_dict[key]
                times = timing_dict[key]
                lens = lens_dict[key]
                best_times = mean_min(best_times, times)
                best_lens = mean_min(best_lens, lens)
                best_succ = mean_max(best_times, succs)
                row_it.append((succs, times, lens))
            for succs, times, lens in row_it:
                # figure out whether anything is best-in-row
                bold_succ = succs and np.allclose(np.mean(succs), best_succ)
                bold_times = times and np.allclose(np.mean(times), best_times)
                bold_lens = lens and np.allclose(np.mean(lens), best_lens)
                succs_str = '%d/%d' % (sum(succs), len(succs))
                if bold_succ:
                    succs_str = '<strong>%s</strong>' % succs_str
                times_str = '-' if not times else '%.3f (±%.3f)' \
                    % (np.mean(times), np.std(times))
                if bold_times:
                    times_str = '<strong>%s</strong>' % times_str
                lens_str = '[-]' if not lens else '[%.3f (±%.3f)]' \
                    % (np.mean(lens), np.std(lens))
                if bold_lens:
                    lens_str = '<strong>%s</strong>' % lens_str
                cell_conts = '<br />'.join([succs_str, times_str, lens_str])
                row_contents += '<td>%s</td>' % cell_conts
            table_rows.append(row_contents)
        all_rows_strs = ('<tr>%s</tr>' % ''.join(row_contents)
                         for row_contents in table_rows)
        table_str = '<table>%s</table>' % '\n'.join(all_rows_strs)
        html_segment = '<h2>%s</h2>\n%s\n<hr/>' % (domain, table_str)
        html_segments.append(html_segment)
    html_inner = '\n'.join(html_segments)
    all_html = """<html>
        <head>
            <title>Results for various planners</title>
            <meta charset="utf-8" />
            <style>
                body {
                    font-family: sans-serif;
                    color: #666;
                }
                table {
                    border-collapse: separate;
                    border-spacing: 1em;
                }
                h1, h2, h3, th {
                    color: #333;
                }
                td {
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <h1>Results for various planners</h1>
            <p>
                Each row represents an instances from a particular domain. Each
                cell shows coverage (out of all runs), mean time for successful
                runs, and mean solution length for successful runs, in that
                order. The timeout was set to %ds.
            </p>
            <hr/>
            %s
        </body>
    </html>""" % (TIMEOUT, html_inner)

    print("Writing results to '%s'" % OUT_FILE)
    with open(OUT_FILE, 'w') as fp:
        fp.write(all_html)


if __name__ == '__main__':
    main()
