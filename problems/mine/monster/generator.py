#!/usr/bin/env python3
"""Generator for Monster problems. See monster.pddl for domain description."""

import argparse
import os


def format_problem(n, name):
    assert n >= 1
    rv = []

    # prelude
    rv = ['(define (problem %s)' % name, '  (:domain monster)']

    # all locations
    locs = [
        [side + '-%d' % i for i in range(1, n)] + [side + '-end']
        for side in ['left', 'right']
    ]
    if n > 1:
        # we don't bother for n == 1 because there are no objects
        rv.extend([
            '  (:objects',
            '    ' + ' '.join(' '.join(l[:-1]) for l in locs),
            '    - location)'
        ])  # yapf: disable

    # start state
    lr_conns = [
        '(conn %s %s)' % (fst, snd)
        for l in locs
        for fst, snd in zip(l, l[1:])
    ]
    # start -> left/right side
    lr_conns += ['(conn start %s)' % l[0] for l in locs]
    # end -> left/right side
    lr_conns += ['(conn %s finish)' % l[-1] for l in locs]
    rv.extend([
        '  (:init (robot-at start)',
        # connections along left and right paths
        '    ' + ' '.join(lr_conns),
        '   )',
    ])  # yapf: disable

    # goal
    rv.append('  (:goal (and (robot-at finish)))')
    rv.append(')')

    return '\n'.join(rv)


parser = argparse.ArgumentParser(
    description='generate a Monster problem of size n')
parser.add_argument('n', type=int, help='points along each path')
parser.add_argument(
    '--dest-dir',
    default=None,
    help='directory to store in (otherwise just print)')


def main():
    args = parser.parse_args()
    name = 'monster-n%i' % args.n

    prob_str = format_problem(args.n, name)
    if args.dest_dir is not None:
        os.makedirs(args.dest_dir, exist_ok=True)
        dest_path = os.path.join(args.dest_dir, name + '.pddl')
        print('Writing to %s' % dest_path)
        with open(dest_path, 'w') as fp:
            print(prob_str, file=fp)
    else:
        print(prob_str)


if __name__ == '__main__':
    main()
