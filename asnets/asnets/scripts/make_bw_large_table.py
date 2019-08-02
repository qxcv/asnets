#!/usr/bin/env python3
"""Make silly table showing distribution over block sizes for extended blocks
world run."""

import os
import re

import click

import pandas as pd

NAME_RE = re.compile(
    # success-blocks-nblk35-seed2107726020-seq42
    r'\b(?P<succ_fail>success|failure)-blocks-nblk(?P<nblk>\d+)(-ntow'
    r'(?P<ntow>\d+))?-seed\d+-seq\d+\b')


@click.command()
@click.option(
    '--directory',
    default='./',
    help='directory with all the blocks results in them (e.g folders of the '
    'form `success-blcoks-nblk50-ntow4-seed…`)')
def main(directory):
    file_names = os.listdir(directory)
    result_dicts = []
    for file_name in file_names:
        match = NAME_RE.match(file_name)
        if not match:
            print("Skipping '%s'" % file_name)
            continue
        groups = match.groupdict()
        success = groups['succ_fail'] == 'success'
        if groups['ntow'] is None:
            ntow = 'R'
        else:
            ntow = int(groups['ntow'])
        data_dict = {
            'successes': success,
            'failures': not success,
            'instances': 1,
            'blocks': int(groups['nblk']),
            'towers': ntow,
        }
        result_dicts.append(data_dict)
    result_frame = pd.DataFrame.from_records(result_dicts)
    grouped_data = result_frame.groupby(['blocks', 'towers'])
    group_counts = grouped_data['instances', 'failures'].sum()
    group_counts = group_counts.astype('int')
    latex_rows = group_counts.T.to_latex()
    print(latex_rows)


if __name__ == '__main__':
    main()
