#!/usr/bin/env python3
"""Make table of coverage and solution costs for ASNets and whatever baselines
you want."""

from argparse import ArgumentParser
import os
import re

from asnets.scripts.solution_time_plot import load_data, add_common_parser_opts

parser = ArgumentParser(
    description="tables of coverage and solution cost for JAIR paper")
parser.add_argument(
    '--assume-det',
    action='store_true',
    default=False,
    help='assume deterministic problem with integer costs')
parser.add_argument(
    '--save', default=None, help='save table to the given file')
add_common_parser_opts(parser)


def main():
    args = parser.parse_args()
    results = load_data(args)
    if args.save:
        dest_dir = os.path.dirname(os.path.abspath(args.save))
        os.makedirs(dest_dir, exist_ok=True)
        with open(args.save, 'w') as fp:
            print_table(results, args.assume_det, file=fp)
    else:
        print_table(results, args.assume_det)


def method_name_comp(method_name):
    # compare first on whether 'asnet' is in the name, then lexicographically
    # on actual name
    is_asnet = 'asnet' in method_name.lower()
    return (int(not is_asnet), method_name)


def simplify_problem_name(prob_name):
    # attempt to simplify a problem name so that it's nicer to disable in the
    # table :)
    gm_match = re.findall(r'^(gold-miner-)target-(\d+x\d+-\d+)-typed-'
                          'gold-miner-testing-target-typed-.+$', prob_name)
    if gm_match:
        return ''.join(gm_match[0])
    mbw_match = re.findall(r'^.+-matching-bw-testing-target-typed-matching-'
                           'bw-target(-n\d+-\d+)-typed$', prob_name)
    if mbw_match:
        return 'mbw-ipc' + mbw_match[0]
    sok_match = re.findall(r'^(sokoban)-.+-testing-target-typed-sokoban-target'
                           '(-n\d+-b\d+-w\d+-\d+)-typed$', prob_name)
    if sok_match:
        return ''.join(sok_match[0])
    park_match = re.findall(r'^(?:parking|testing|target|typed|-)+(-c\d+-\d+)'
                            r'-typed$', prob_name)
    if park_match:
        return 'parking' + park_match[0]
    return prob_name


def first_word(string):
    # get first word of string
    parts = re.split(r'[^a-zA-Z0-9]', string)
    return parts[0]


def print_table(results, assume_det, file=None):
    method_names = set()
    prob_names = set()
    prob_sizes = {}
    # maps (method, prob) tuple to cell contents
    problem_data = {}
    # TODO: bold cells in each row based on cost (for deterministic problems)
    # or coverage (for probabilistic problems). Should be really easy for
    # readers to scan the table and see what the fuck is going on.
    for method_label, data in results:
        summary = data['summary']
        names = data['eval_names']
        names = list(map(simplify_problem_name, names))
        # sometimes names longer than sizes for some reason :/
        sizes = data['eval_sizes']
        method_names.add(method_label)
        for prob_idx, prob_name in enumerate(names[:len(sizes)]):
            prob_names.add(prob_name)
            prob_sizes[prob_name] = sizes[prob_idx]
            num_solved, total = summary['cov_pairs'][prob_idx]
            cost_mu = summary['cost_mean'][prob_idx]
            cost_ci = summary['cost_ci'][prob_idx]
            if assume_det:
                # this is deterministic, so just print cost (and optionally
                # fraction reaching the goal, if it's nonzero)
                if num_solved == 0:
                    cell_contents = '-'
                else:
                    if cost_mu == int(cost_mu):
                        # print whole numbers the pretty way
                        cell_contents = '%d' % cost_mu
                    else:
                        cell_contents = '%.1f' % cost_mu
                    if total != 1:
                        cell_contents += ' (%d/%d)' % (num_solved, total)
            else:
                # maybe probabilistic, so need CI for cost, too
                if num_solved == 0:
                    cell_contents = '-'
                elif num_solved == 1:
                    cell_contents = r'\makecell{%d/%d \\ (%.1f)}' \
                                    % (num_solved, total, cost_mu)
                elif not cost_ci:
                    cell_contents = r'\makecell{%d/%d \\ (%.1f $\pm$ 0)}' \
                                    % (num_solved, total, cost_mu)
                else:
                    cell_contents = r"\makecell{%d/%d \\ (%.1f $\pm$ %.1f)}" \
                                    % (num_solved, total, cost_mu, cost_ci)
            # time_mu = summary['time_mean'][prob_idx]
            # time_ci = summary['time_ci'][prob_idx]
            # cell_contents += ' [T] %s (pm %s)' % (time_mu, time_ci)
            problem_data[(method_label, prob_name)] = cell_contents
    method_names = sorted(method_names, key=method_name_comp)
    print('%', ' & '.join(method_names), r'\\', file=file)
    prob_names = sorted(prob_names, key=lambda k: (prob_sizes[k], k))
    for prob_name in prob_names:
        print(prob_name + r' &\wskip', file=file)
        # now print results for each method with a preceding " & ", inserting
        # \wskip markers between columns for different methods
        method_names_pn = method_names[1:] + [None]
        assert len(method_names_pn) >= len(method_names)
        for meth_name, next_meth_name in zip(method_names, method_names_pn):
            key = (meth_name, prob_name)
            contents = problem_data.get(key, '-')
            print('  & %s' % contents, file=file, end='')
            # break if the method names have different first word (e.g LRTDP vs
            # SSIPP)
            col_break = next_meth_name is not None and \
                first_word(meth_name) != first_word(next_meth_name)
            if col_break:
                print(r' &\wskip', file=file, end='')
            print(' %% %s' % meth_name, file=file)
        # linebreak at end of row
        print(r'  \\', file=file)


if __name__ == '__main__':
    main()
