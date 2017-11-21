#!/usr/bin/env python3
"""Generator for CosaNostra Pizza problems."""

import argparse
import os


def format_problem(n, name, b_steps=0):
    assert n > 0
    rv = []

    # prelude
    rv = ['(define (problem %s)' % name, '  (:domain cosanostra)']

    # all checkpoints
    rv.extend([
        '  (:objects ',
        '    %s - toll-booth' % ' '.join('b%d' % i for i in range(n)),
    ])

    if b_steps > 0:
        rv.extend(['    %s sG - b-step' % ' '.join('s%d' % i for i in range(b_steps))])

    rv.extend(['    shop home - open-intersection)'])  # yapf: disable

    # start state
    tb_roads = ' '.join('(road b{0} b{1}) (road b{1} b{0})'.format(i, j)
                     for i, j in zip(range(n), range(1, n)))
    rv.extend([
        '  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)',
        '    (road shop b0) (road b0 shop)',
        '    (road home b{0}) (road b{0} home)'.format(n-1),
        '    ' + tb_roads,
    ])  # yapf: disable

    # Extra chain of bureaucracy
    if b_steps > 0:
        bureaucracy = ' '.join('(b_next s{0} s{1})'.format(i, j)
                     for i, j in zip(range(b_steps), range(1, b_steps)))
        rv.extend([
            '    (b_next s{0} sG)'.format(b_steps-1),
            '    ' + bureaucracy,
            '    (bureaucracy s0)',
        ])
    rv.append('  )')
    # goal
    rv.append('  (:goal (and (pizza-at home) (deliverator-at shop))))')

    return '\n'.join(rv)


parser = argparse.ArgumentParser(
    description='generate random CosaNostra problems')
parser.add_argument('n', type=int, help='vertices in graph')
parser.add_argument('-b', type=int, help='size of the bureaucracy chain', default=0)
parser.add_argument(
    '--dest-dir',
    default=None,
    help='directory to store in (otherwise just print)')


def main():
    args = parser.parse_args()
    name = 'cosanostra-n%i' % args.n

    if args.b > 0:
        name += '-b%i' % args.b
    prob_str = format_problem(args.n, name, args.b)
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
