#!/usr/bin/env python3
"""Make a table of peak coverages for different domains using a bunch of
ablated ASNet runs. Much simpler than other tables, and designed only to give
an idea of how well ASNets do across the entire set of test domains with
various ablations."""

import argparse

import pandas as pd

from asnets.scripts.solution_time_plot import add_common_parser_opts
from asnets.scripts.cumulative_solved_plot import load_labels_frames

# What script usage syntax do I want? Maybe I can just do the normal thing, but
# with some stuff repeated? Then I could use name_expt to pull out a correct
# domain description & distinguish between problems from different domains. In
# fact, I could also pull out name_arch from the JSON files. So, calling this
# script is going to look something like:
#
#   python -m asnets.scripts.ablation_coverage_plot \
#       "ASNets":/path/to/asnets-dom1.json \
#       "ASNets (no LM)":/path/to/asnets-no-lm-dom1.json \
#       "ASNets":/path/to/asnets-dom2.json \
#       "ASNets (no LM)":/path/to/asnets-no-lm-dom2.json
#
# No domain names; they'll just be inferred from the module names. I can
# probably also add experiment name aliasing options if I want to.


def main(args):
    all_labels, data_frame = load_labels_frames(args)
    # num_problems = len(data_frame["problem"].unique())
    methods = data_frame["method"].unique()
    # TODO: order the domains [probabilistic -> deterministic]
    domains = data_frame['name_expt'].unique()
    dom_alias_dict = dict(args.dom_alias)
    # the baselines don't include name_expt, but the ASNets do
    assert 'EMPTY' not in domains, \
        "some domain names are not present as name_expt in the .json files"

    # remove all the problems that exceed the max size for their domain
    max_sizes = {
        dom_name: int(max_size_s)
        for dom_name, max_size_s in args.dom_max_size
    }

    def size_filter(row):
        dom = row['name_expt']
        size = row['problem_size']
        return dom not in max_sizes or size <= max_sizes[dom]

    valid_mask = data_frame.apply(size_filter, axis=1)
    data_frame = data_frame[valid_mask]

    # now group problems by combination of method name & domain, then compute
    # cumulative coverage
    method_name_groups = data_frame.groupby(['method', 'name_expt'])

    def count_prob_coverage(sub_frame):
        sf_groups = sub_frame.groupby('problem')['goal_reached']
        # TODO: might need to convert the result to float instead of int or
        # bool or whatever it is
        return sf_groups.mean()

    prob_cumul_coverage = method_name_groups.apply(count_prob_coverage)
    sum_coverage = prob_cumul_coverage.sum(level=[0, 1])
    num_probs = method_name_groups['problem'].nunique()
    # can use this command to join everything together, if needed:
    # results_table = pd.concat([sum_coverage, num_probs], axis=1)

    # now print table
    print(r'\begin{tabular}{%s}' % ('l' + 'c' * len(domains)))
    print(r'  \toprule')
    fancy_dom_names = [dom_alias_dict.get(ident, ident) for ident in domains]
    print('  Ablation &', ' & '.join(fancy_dom_names), end="\\\\\n")
    print(r'  \midrule')
    for method_name in methods:
        print(f'  {method_name}', end='')
        for domain_ident in domains:
            key = (method_name, domain_ident)
            this_sum_cov = sum_coverage[key]
            this_num_probs = num_probs[key]
            print(' & %.1f/%s' % (this_sum_cov, this_num_probs), end='')
        print(r'\\')
    print(r'  \bottomrule')
    print(r'\end{tabular}')


parser = argparse.ArgumentParser(
    description="Make table of peak coverages for some ablated ASNets")
parser.add_argument(
    '--dom-alias',
    nargs=2,
    action='append',
    default=[],
    help='replace domain name (first) with some other name (second) in output')
parser.add_argument(
    '--dom-max-size',
    nargs=2,
    action='append',
    default=[],
    help='max size of problems in domain')
add_common_parser_opts(parser)

if __name__ == '__main__':
    main(parser.parse_args())
