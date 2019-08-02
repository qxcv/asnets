#!/usr/bin/env python3
"""Complete generator for deterministic blocksworld problems. Supports options
that allow it to generate a mixed set of both challenging and random
problems."""

import argparse
import os
import random
import subprocess

# Domain taken from pyperplan. I don't know where they got it from.
# (note that my choice of line escapes omits the newline after the opening
# triple quote but retains the newline before the closing triple quote; that is
# intentional)
DOMAIN = """\
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 4 Op-blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (ontable ?x - block)
    (clear ?x - block)
    (handempty)
    (holding ?x - block))

  (:action pick-up
   :parameters (?x - block)
   :precondition (and (clear ?x) (ontable ?x) (handempty))
   :effect (and
     (not (ontable ?x))
     (not (clear ?x))
     (not (handempty))
     (holding ?x)))

  (:action put-down
   :parameters (?x - block)
   :precondition (holding ?x)
   :effect (and
     (not (holding ?x))
     (clear ?x)
     (handempty)
     (ontable ?x)))

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and
      (not (holding ?x))
      (not (clear ?y))
      (clear ?x)
      (handempty)
      (on ?x ?y)))

  (:action unstack
   :parameters (?x - block ?y - block)
   :precondition (and (on ?x ?y) (clear ?x) (handempty))
   :effect (and
      (holding ?x)
      (clear ?y)
      (not (clear ?x))
      (not (handempty))
      (not (on ?x ?y)))))
"""
PROBLEM_TEMPLATE = """\
(define (problem %(name)s)
    (:domain blocks)
    (:objects %(blocknames)s - block)
    (:init %(init)s)
    (:goal (and %(goal)s)))
"""
if '__file__' in globals():
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))
else:
    THIS_DIR = '.'


def format_state(below_arr, add_hand=True):
    parts = []
    if add_hand:
        parts.append('(handempty)')
    for top, bottom in enumerate(below_arr, start=1):
        assert isinstance(bottom, int) and 0 <= bottom <= len(below_arr)
        if bottom == 0:
            parts.append('(ontable b%d)' % top)
        else:
            parts.append('(on b%d b%d)' % (top, bottom))
    nblocks = len(below_arr)
    clears = sorted(set(range(1, nblocks + 1)) - set(below_arr))
    for clear in clears:
        parts.append('(clear b%d)' % clear)
    return ' '.join(parts)


def format_problem(name, init_state, goal_state):
    assert len(init_state) == len(goal_state)
    blocknames = ' '.join('b%d' % d for d in range(1, len(init_state) + 1))
    full_pddl = PROBLEM_TEMPLATE % dict(
        name=name,
        blocknames=blocknames,
        init=format_state(init_state),
        goal=format_state(goal_state), )
    return full_pddl


def run_bwkstates(nblocks, seed, *, ntowers=None, nstates=1):
    assert isinstance(nblocks, int) and isinstance(seed, int)
    executable = os.path.join(THIS_DIR, 'bwkstates')
    cmdline = [
        executable, '-n', str(nblocks), '-r', str(seed), '-s', str(nstates)
    ]
    if ntowers is not None:
        ntowers_s = str(ntowers)
        cmdline.extend(['-i', str(ntowers_s), '-g', str(ntowers_s)])
    proc_info = subprocess.run(
        cmdline, check=True, universal_newlines=True, stdout=subprocess.PIPE)
    result = proc_info.stdout.splitlines()
    states = []
    assert len(result) >= 1 and result[-1] == '0'
    for first_line, second_line in zip(result[::2], result[1::2]):
        assert first_line.startswith(' ') and second_line.startswith(' ')
        nblocks = int(first_line[1:])
        str_below_arr = second_line[1:].split()
        assert nblocks == len(str_below_arr)
        below_arr = list(map(int, str_below_arr))
        states.append(below_arr)
    assert len(states) == nstates
    return states


def main():
    # first figure out arguments, setting defaults as necessary
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--write-domain',
        default=False,
        action='store_true',
        help='if supplied, a domain will be written to `domain.pddl` in the '
        '`dest` directory')
    parser.add_argument(
        '--ntowers',
        type=int,
        default=None,
        help='number of towers in state (by default determined by bwstates)')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed (uses random.randint by default)')
    parser.add_argument(
        '--nprobs',
        default=1,
        type=int,
        help='number of problems to generate (default 1)')
    parser.add_argument('nblocks', type=int, help='number of blocks in state')
    parser.add_argument('destdir', type=str, help='destination directory')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    # set up output dir
    os.makedirs(args.destdir, exist_ok=True)
    if args.write_domain:
        dom_path = os.path.join(args.destdir, 'domain.pddl')
        print('Writing domain to %s' % dom_path)
        with open(dom_path, 'w') as dom_fp:
            dom_fp.write(DOMAIN)

    # now generate PDDL
    all_states = run_bwkstates(
        nblocks=args.nblocks,
        seed=args.seed,
        ntowers=args.ntowers,
        nstates=2 * args.nprobs)
    inits, goals = all_states[::2], all_states[1::2]
    for seq in range(args.nprobs):
        init = inits[seq]
        goal = goals[seq]
        name_parts = ['blocks', 'nblk%d' % args.nblocks]
        if args.ntowers is not None:
            name_parts.append('ntow%d' % args.ntowers)
        name_parts.extend(['seed%d' % args.seed, 'seq%d' % seq])
        name = '-'.join(name_parts)
        problem_pddl = format_problem(name, init, goal)
        dest_fn = 'prob-blocks-%s.pddl' % name
        dest_path = os.path.join(args.destdir, dest_fn)

        # finally write to output directory
        print('Writing problem %d/%d to %s' %
              (seq + 1, args.nprobs, dest_path))
        with open(dest_path, 'w') as fp:
            fp.write(problem_pddl)


if __name__ == '__main__':
    main()
