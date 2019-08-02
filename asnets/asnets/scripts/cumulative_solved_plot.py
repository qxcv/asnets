#!/usr/bin/env python3
"""A tool to plot cumulative time taken for each planner to solve a collection
of problems. Really needs to be renamed to avoid confusion with
solution_time_plot.py."""

from argparse import ArgumentParser
from collections import OrderedDict
from json import load

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from asnets.scripts.solution_time_plot import add_common_parser_opts

parser = ArgumentParser(
    description="plots of total running time for ICAPS/JFPDA slides (and "
    "maybe JAIR too)")
parser.add_argument('--save',
                    metavar='PATH',
                    type=str,
                    default=None,
                    help="destination file for graph")
parser.add_argument('--no-legend',
                    dest='add_legend',
                    default="brief",
                    action='store_false',
                    help='disable legend')
parser.add_argument('--dims',
                    nargs=2,
                    type=float,
                    metavar=('WIDTH', 'HEIGHT'),
                    default=[7, 3],
                    help="dimensions (in inches) for saved plot")
parser.add_argument('--xmax',
                    type=int,
                    help='maximum time to show along x-axis')
parser.add_argument('--title', help='title for the plot')
parser.add_argument(
    '--presentation',
    default=False,
    action='store_true',
    help='use typeface/display choices appropriate for presentation (rather '
    'than printed work)')

add_common_parser_opts(parser)


def _load_inner_df(expt_str, args):
    """Load single experiment outcome as dataframe."""
    try:
        label, path = expt_str.split(':', 1)
    except ValueError as e:
        print('Could not parse label:path pair "%s"' % expt_str)
        raise e

    # this is for data that we don't have because one method took forever to
    # solve anything :(
    no_data = path == 'EMPTY'
    if no_data:
        data = {'eval_names': [], 'eval_sizes': [], 'eval_runs': []}
    else:
        # load data
        with open(path, 'r') as fp:
            data = load(fp)

    # sometimes we have more names than sizes for some reasons (probably
    # collate_data is broken)
    num_runs_set = set(
        map(len, [data['eval_sizes'], data['eval_runs'], data['eval_names']]))
    assert len(num_runs_set), "inconsistent sizes (%s)" % (num_runs_set, )

    # this is used for ASNets
    train_time = data.get('train_time', 0)

    # These are sometimes used by the calling code. name_arch is the name of
    # the architecture module (e.g. actprop_2l), and name_expt is the name of
    # the experiment module (e.g. ex_blocksworld).
    name_arch = data.get('name_arch', 'EMPTY')
    name_expt = data.get('name_expt', 'EMPTY')

    # record format:
    # {
    #   "problem": <problem-name>,
    #   "problem_size": <problem-size>,
    #   "method": <algorithm-name>,
    #   "goal_reached": <goal-reached>,
    #   "cost": <cost-or-deadend-cost>,
    #   "time": <time-or-timeout>,
    #   "time_raw": <ray-time-maybe-none>,
    #   "run_seq_num": <run-num-in-sequence>,
    # }
    # We'll make a DataFrame out of those!
    records = []
    for name, size, data_dict in zip(data['eval_names'], data['eval_sizes'],
                                     data['eval_runs']):
        for seq_num, (goal_reached, cost, time) in enumerate(
                zip(data_dict['goal_reached'], data_dict['cost'],
                    data_dict['time'])):
            if time is None:
                time_or_timeout = args.timeout
            else:
                # always add in training time
                time_or_timeout = time + train_time
            record = {
                "problem": name,
                "problem_size": size,
                "method": label,
                "goal_reached": goal_reached,
                "cost": cost,
                "time": time_or_timeout,
                # use nans instead of None to hint to Pandas that this series
                # should be float, and not object
                "time_raw": time if time is not None else float('nan'),
                "train_time": train_time,
                "run_seq_num": seq_num,
                "name_arch": name_arch,
                "name_expt": name_expt,
            }
            records.append(record)

    frame = pd.DataFrame.from_records(records)

    return label, frame


def load_labels_frames(args):
    """Load an entire collection of experiments as a big DataFrame. Also return
    labels, which I suspect I'll need at some point (I could get them out of
    the DataFrame, but that'll only work for the methods where I have at least
    some data)."""
    frames = []
    labels = []
    for label_path in args.jsons:
        label, frame = _load_inner_df(label_path, args)
        labels.append(label)
        frames.append(frame)
    mega_frame = pd.concat(frames)
    return labels, mega_frame


def main():
    args = parser.parse_args()

    if args.presentation:
        sns.set(context='talk', style='whitegrid')
        matplotlib.rcParams.update({
            'font.family': 'sans',
            'font.serif': 'sans',
            'pgf.rcfonts': False,
            'pgf.texsystem': 'pdflatex',
            # 'xtick.labelsize': 'x-small',
            # 'ytick.labelsize': 'x-small',
            'legend.fontsize': 10,
            # 'axes.labelsize': 'x-small',
            # 'axes.titlesize': 'medium'
        })
    else:
        # Make the plot look like LaTeX
        sns.set(context='paper', style='whitegrid')
        matplotlib.rcParams.update({
            'font.family': 'CMU Serif',
            'font.serif': 'CMU Serif',
            'pgf.rcfonts': False,
            'pgf.texsystem': 'pdflatex',
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
            'legend.fontsize': 6.5,
            'axes.labelsize': 'small',
            'axes.titlesize': 'medium',
            # ensures that we have only type 42/type 1 fonts, and *no* type
            # 3---important for AAAI (type 42 = TrueType)
            # 'pdf.fonttype': 42,
            # 'ps.fonttype': 42,
            'text.usetex': True,
        })

    all_labels, data_frame = load_labels_frames(args)
    size_mask = (data_frame['problem_size'] >= args.min_size) \
        & (data_frame['problem_size'] <= args.max_size)
    data_frame = data_frame[size_mask]

    # Now we plot cumulative time
    # record format:
    # {
    #    "method":
    #    "ctime":
    #    "coverage":
    # }
    cumul_records = []
    max_covs = OrderedDict()
    max_ctime = 0.0
    for method in data_frame['method'].unique():
        selection = data_frame[data_frame['method'] == method]
        grouped = selection.groupby('problem')
        sol_times = grouped['time_raw'].mean().to_numpy()
        sol_times[~np.isfinite(sol_times)] = 0.0
        sol_coverages = grouped['goal_reached'].mean().to_numpy()\
            .astype('float')
        train_time = grouped['train_time'].mean().mean()
        # append empty initial record
        cumul_records.append({
            "method": method,
            "coverage": 0,
            "ctime": 0,
        })
        if train_time > 0:
            # record for train time is important
            cumul_records.append({
                "method": method,
                "coverage": 0,
                "ctime": train_time,
            })
        # want to sort by gradient, which is sol_coverages / sol_times (with
        # 0/0=0)
        gradients = np.divide(sol_coverages,
                              sol_times,
                              out=np.zeros_like(sol_coverages),
                              where=sol_times > 0)
        correct_order = np.argsort(gradients)[::-1]
        ctime = train_time
        ccov = 0
        for prob_num in correct_order:
            this_cov = sol_coverages[prob_num]
            if this_cov == 0:
                break
            ccov += this_cov
            this_time = sol_times[prob_num]
            ctime += this_time
            cumul_records.append({
                "method": method,
                "coverage": ccov,
                "ctime": ctime,
            })
        max_covs[method] = ccov
        max_ctime = max(ctime, max_ctime)

    right_point = max(max_ctime, args.xmax or 0)
    for method, max_cov in max_covs.items():
        cumul_records.append({
            "method": method,
            "coverage": max_cov,
            "ctime": right_point,
        })

    ctime_frame = pd.DataFrame.from_records(cumul_records)
    ctime_frame['hours'] = ctime_frame['ctime'] / 3600.0
    ctime_frame['Method'] = ctime_frame['method']  # capitalise legend title :)
    sns.lineplot(data=ctime_frame,
                 x="hours",
                 y="coverage",
                 hue="Method",
                 drawstyle='steps-pre',
                 legend=args.add_legend and 'brief')
    num_problems = len(data_frame['problem'].unique())
    plt.axhline(num_problems, ls='--', color='maroon')
    plt.xlabel("Cumulative time (hours)")
    plt.ylabel("Cumulative coverage")
    max_secs = args.xmax or ctime_frame['ctime'].max()
    max_h = max_secs / 3600.0
    plt.xlim(left=-0.025 * max_h, right=max_h)
    plt.ylim(bottom=-num_problems * 0.01, top=num_problems * 1.05)

    # title
    plt.title(args.title)

    # save or show
    if args.save is None:
        plt.show()
    else:
        print('Saving figure to', args.save)
        plt.gcf().set_size_inches(args.dims)
        plt.tight_layout()
        plt.savefig(
            args.save,
            bbox_inches='tight',
            dpi=400,
            # bbox_extra_artists=[legend],
            transparent=True)


if __name__ == '__main__':
    main()
