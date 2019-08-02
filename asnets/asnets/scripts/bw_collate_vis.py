#!/usr/bin/env python3
"""Collate & visualise deterministic blocksworld results. Hopefully this will
tell us (1) how often we're succeeding, and (2) what plans we're actually
creating."""

import argparse
import json
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        # silently ignore the progress bar
        yield from it

# pulls the problem name out of PDDL declaration of the form "(define (problem
# <name-here>))"
PROBLEM_DEF_RE = re.compile(
    r'(^|\n)\s*\(define\s*\(problem\s+(?P<prob_name>[^);\s]+)\s*\)')


def read_pddl_prob_name(pddl_path):
    """Use a regular expression to figure out the name of the problem declared
    in the PDDL file at the given path.

    Corner cases: returns None if no problem name can be found; returns only
    first problem name if more than one problem is declared in a file."""
    with open(pddl_path, 'r') as fp:
        pddl_contents = fp.read()
    prob_match = PROBLEM_DEF_RE.match(pddl_contents)
    if prob_match is None:
        return None
    prob_name = prob_match.groupdict()['prob_name']
    return prob_name


def filepath_stream(root_dir):
    """Explores all files in the hierarchy below `root_dir`, and yields their
    paths one-at-time."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            yield filepath


def index_pddls(pddl_dir):
    """Discover all PDDL files in `pddl_dir` (using file extensions) & return a
    mapping of [problem name]->[path for .pddl file]."""
    # `warned` contains problem names for which we have warned about
    # duplicates; helps avoid duplicated duplicate warnings
    warned = set()
    pddl_suffixes = ['.pddl', '.ppddl']
    name_path_mapping = {}
    for filepath in filepath_stream(pddl_dir):
        # check that it has appropriate PDDL file extension
        fp_lower = filepath.lower()
        valid_suffix = any(fp_lower.endswith(suff) for suff in pddl_suffixes)
        if not valid_suffix:
            continue
        # if we have appropriate PDDL file extension, try to read problem
        # definition from the file
        # FIXME: this will break when several problem definitions are present
        # in the same file :(
        prob_name = read_pddl_prob_name(filepath)
        if prob_name is None:
            # this is probably a domain, so skip
            continue
        # warn about problem names that appear in more than one file
        if prob_name in name_path_mapping and prob_name not in warned:
            print(
                "Duplicate problem name '{prob_name}', refers to both "
                "'{first_path}' and '{second_path}' (won't warn again)".format(
                    prob_name=prob_name,
                    first_path=name_path_mapping[prob_name],
                    second_path=filepath),
                file=sys.stderr)
            warned.add(prob_name)
        name_path_mapping[prob_name] = filepath
    return name_path_mapping


def discover_result_jsons(exp_subdir):
    """Discover (hopefully) all `results.json` files in the FS hierarchy below
    `exp_subdir`. Returns list of (JSON path, parsed JSON contentAbout 257,000
    results (0.46 seconds) s) pairs."""
    result_jsons = []
    for run_dir in os.listdir(exp_subdir):
        if not run_dir.startswith('P['):
            continue
        result_json_path = os.path.join(exp_subdir, run_dir, 'results.json')
        if not os.path.isfile(result_json_path):
            continue
        with open(result_json_path, 'r') as json_fp:
            result_json_parsed = json.load(json_fp)
        result_jsons.append((result_json_path, result_json_parsed))
    result_jsons = sorted(result_jsons, key=lambda t: t[0])
    return result_jsons


def run_pbw_solve(prob_name, pddl_path, asnet_plan):
    """Runs blocksworld visualiser to (1) visualise & verify given ASNet plan,
    and (2) get baseline plan costs. Returns a [method name]->[info dict]
    mapping, where [info dict] is a dictionary with "prob_name", "cost", and
    "visualisation" keys."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as plan_fp:
        # write ASNet plan to temp file & get path of result
        plan_fp.writelines(act + '\n' for act in asnet_plan)
        plan_fp.flush()
        manual_plan_path = plan_fp.name

        # now run pbw_solve.py & store result
        pbw_solve_cmd = [
            'python', '-m', 'asnets.scripts.pbw_solve', '--strategy', 'US',
            '--strategy', 'MANUAL', '--strategy', 'GN1', '--strategy', 'GN2',
            '--strategy', 'OPTIMAL', '--manual-plan', manual_plan_path,
            '--allow-invalid-plans', '--pretty-mode', 'full-plan', 'det',
            pddl_path,
        ]
        cmd_result = subprocess.run(
            pbw_solve_cmd,
            check=True,
            universal_newlines=True,
            stdout=subprocess.PIPE)
    # here's the structure we get:
    #
    # RESULT|File <basename>|Method <GN1 etc.>|Cost <cost>|Actions <#actions>
    # START PLAN VIS
    # <visualiation goes here>
    # END PLAN VIS
    rev_lines = cmd_result.stdout.splitlines()[::-1]
    meth_data = {}
    while len(rev_lines) > 0:
        # first parse single metadata line
        res_line = rev_lines.pop()
        assert res_line.startswith(
            'RESULT|'), "expected 'RESULT|*' but got '%s'" % (res_line, )
        part_dict = dict(l.split(maxsplit=1) for l in res_line.split('|')[1:])
        method = part_dict['Method']
        assert part_dict['Cost'].isdigit(), \
            "plan cost '%s' is not an integer" % (part_dict['Cost'], )
        cost = int(part_dict['Cost'])
        is_valid_plan = part_dict['Valid'] == 'True'

        # now parse visualisation
        vis_line = rev_lines.pop()
        assert vis_line == 'START PLAN VIS', \
            "vis_line '%s' is wrong" % (vis_line, )
        vis_lines = [vis_line]
        while vis_line != 'END PLAN VIS':
            vis_line = rev_lines.pop()
            vis_lines.append(vis_line)
        # [1:-1] removes the "{START,END} PLAN VIS" lines
        visualisation = '\n'.join(vis_lines[1:-1])

        # finally store
        meth_data[method] = {
            'cost': cost,
            'visualisation': visualisation,
            'prob_name': prob_name,
            'is_valid_plan': is_valid_plan,
        }

    return meth_data


def string_format_table(rows, header_rows=1, header_cols=1):
    """Format a table (i.e. nested list) of Python values as text."""
    # convert all rows to string and record widths
    new_rows = []
    col_widths = []
    for row in rows:
        # convert to string
        new_row = [str(r) for r in row]
        new_rows.append(new_row)

        # record width
        for idx in range(len(new_row)):
            this_col_width = len(new_row[idx])
            if len(col_widths) <= idx:
                col_widths.append(this_col_width)
            else:
                col_widths[idx] = max(col_widths[idx], this_col_width)

    # set up out buffer
    out_buf = io.StringIO()

    def emit(*args, **kwargs):
        print(*args, file=out_buf, end='', **kwargs)

    # now format table
    rows = new_rows
    tot_width = sum(cw + 2 for cw in col_widths) + header_cols
    for row_num, row in enumerate(rows, start=1):
        for col_num, col in enumerate(row, start=1):
            emit(' ')
            col_width = col_widths[col_num - 1]
            emit(col.rjust(col_width))
            emit(' ')
            if col_num <= header_cols:
                # header cols separated from rest by bar
                emit('|')
        if row_num <= header_rows:
            emit('\n')
            emit('-' * tot_width)
        if row_num < len(rows):
            emit('\n')

    return out_buf.getvalue()


def filename_sane_replace(some_string):
    """Replace non-alphanumeric characters in a strong with dashes. Should make
    it safe to write out to any sane (i.e. non-MS) filesystem."""
    return "".join((c if c.isalnum() else '-') for c in some_string)


def write_individual_summary(pbw_solve_result, out_dir):
    """Create a summary report for an individual problem. At the moment this
    only contains a visualisation of the ASNet-generated plan used to solve the
    problem."""
    prob_name = pbw_solve_result['MANUAL']['prob_name']
    out_fn = 'plan-' + filename_sane_replace(prob_name) + ".txt"
    if not pbw_solve_result['MANUAL']['is_valid_plan']:
        # prepend "failed" so that we can spot the (usually rare) failed plans
        # easily
        out_fn = 'failed-' + out_fn
    out_path = os.path.join(out_dir, out_fn)
    with open(out_path, 'w') as out_fp:
        print("ASNet-generated plan to solve '%s'" % prob_name, file=out_fp)
        print('', file=out_fp)
        print(pbw_solve_result['MANUAL']['visualisation'], file=out_fp)


def _count_problems(run_dir):
    num_problems = '???'
    with open(os.path.join(run_dir, 'run-info', 'stdout')) as stdout_fp:
        # figure out how many problems we trained on
        for line in stdout_fp:
            if line.startswith('Loading problems '):
                num_problems = line.count(',') + 1
                break
    return num_problems


def write_overall_summary(all_results, train_run_data, out_dir):
    """Write a summary for results on a set of problems. The summary includes a
    table indicating how many instances of each size were solved & how long the
    solution length was on each problem.."""
    # How to stringify a table (with no trailing newline---be warned!)
    # string_table = string_format_table(rows, header_rows=1, header_cols=1)
    out_path = os.path.join(out_dir, 'all.txt')
    num_layers = train_run_data['num_layers']
    train_seconds = train_run_data['elapsed_opt_time']
    train_hours = train_seconds / 3600.0
    num_eval = len(all_results)
    num_train_problems = _count_problems(train_run_data['run_dir'])
    prelude_text = "Plan cost summary for plans produced for %s evaluation " \
                   "problems (all were solved by ASNets!). Model was a " \
                   "%s-layer ASNet trained on %s training problems for %.3f " \
                   "hours. Consult plan-<problem>.txt files in this " \
                   "directory for ASNet plan visualisations." \
                   % (num_eval, num_layers, num_train_problems, train_hours)
    plan_length_table = [['Problem', 'US', 'ASNet', 'GN1', 'GN2', 'OPTIMAL']]
    for result in all_results:
        new_row = [
            result['MANUAL']['prob_name'],
            result['US']['cost'],
            result['MANUAL']['cost'],
            result['GN1']['cost'],
            result['GN2']['cost'],
            result['OPTIMAL']['cost'],
        ]
        plan_length_table.append(new_row)
    plan_length_strtable = string_format_table(
        plan_length_table, header_cols=2)
    with open(out_path, 'w') as out_fp:
        print(textwrap.fill(prelude_text), file=out_fp)
        print(plan_length_strtable, file=out_fp)


parser = argparse.ArgumentParser(
    description='collate results for a set of deterministic blocksworld '
    'problems')
parser.add_argument(
    '--out-dir', default='./bw-collated/', help='directory to write output to')
parser.add_argument(
    'pddl_dir',
    metavar='PDDL-DIR',
    help='path to directory with original PDDL problem files')
parser.add_argument(
    'result_dir',
    metavar='RESULT-DIR',
    help='result directory (will probably be a subdir of experiment-results/)')


def main():
    args = parser.parse_args()
    pddl_index = index_pddls(args.pddl_dir)
    print("Discovered %d PDDL problems in '%s'" %
          (len(pddl_index), args.pddl_dir))
    result_jsons = discover_result_jsons(args.result_dir)
    print("Discovered %d results.json files in '%s'" % (len(result_jsons),
                                                        args.result_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    # train_run_data stores raw .json run data from training run
    train_run_data = None
    # all_results stores all other results in different format produced by
    # run_pbw_solve()
    all_results = []
    for result_json_path, result_data in tqdm(result_jsons):
        # Relevant parts of results.json structure:
        # {
        #  "problem": "blocks-nblk25-seed1428122178-seq6",
        #  "all_goal_reached": [true, true],
        #  "trial_paths": [["unstack b9 b18", …, "GOAL! :D"],
        #                  ["unstack b9 b18", …, "GOAL! :D"]],
        #  "all_costs": [78, 78],
        #  "all_exec_times": [28.005, 27.266],
        #  …
        # }

        if not result_data["no_train"]:
            # this is a training instance, so just copy out info & continue
            assert train_run_data is None, "duplicate train run data found!"
            train_run_data = result_data
            train_run_data['run_dir'] = os.path.dirname(result_json_path)
            continue

        # otherwise this is a test instance, so continue as usual
        trial_paths_uniq = {tuple(tp) for tp in result_data['trial_paths']}
        if len(trial_paths_uniq) > 1:
            # warn user that there are several paths
            print("WARNING: %d unique paths in '%s'; will only use the first" %
                  (len(trial_paths_uniq), result_json_path))
        asnet_plan = result_data['trial_paths'][0][:-1]
        prob_name = result_data['problem']
        pddl_path = pddl_index[prob_name]
        pbw_solve_result = run_pbw_solve(prob_name, pddl_path, asnet_plan)
        all_results.append(pbw_solve_result)

        write_individual_summary(pbw_solve_result, args.out_dir)

    # now we make a pretty aggregated result? IDK.
    assert train_run_data is not None, \
        "didn't find data from training run, can't write overall summary"
    write_overall_summary(all_results, train_run_data, args.out_dir)


if __name__ == '__main__':
    main()
