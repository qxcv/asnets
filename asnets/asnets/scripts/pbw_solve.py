#!/usr/bin/env python3
"""Solve Blocks World (BW) or Probabilistic BW (PBW) instances either using
John Slaney's CGI tool, or using a given plan (deterministic only). Can
pretty-print the plan if desired."""

import argparse
import enum
import json
import os
import re
import string
import warnings

import requests

from asnets.pddl_utils import HList, parse_sexprs
from asnets.py_utils import remove_cycles

# I think this means 'get the full plan'
SCRIPT_VERB = 3
# is another mode possible? IDK
SCRIPT_MODE = 'SOLVE'


class SolutionType(enum.IntEnum):
    US = 0
    GN1 = 1
    GN2 = 2
    OPTIMAL = 3
    MANUAL = 4


VALID_STRATS = [s for s in dir(SolutionType) if s.isupper()]


class Action:
    type = None


# note to self: we only need two action classes (not four) because in BW a
# "pick up that block from the table or another block" action must always be
# followed by a "put that same block onto the table or onto another block"
# action.


class Unstack(Action):
    # move a block (that could be sitting on a table or on another block) back
    # onto the table
    def __init__(self, block):
        self.type = 'unstack'
        self.block = block

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.block)

    def __str__(self):
        return 'Unstack %s onto table' % (self.block, )

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return other.type == self.type and other.block == self.block

    def __hash__(self):
        return hash(self.type) ^ hash(self.block)


class Stack(Action):
    # move a block (that could be sitting anywhere, again) onto another block
    def __init__(self, block, dest):
        self.type = 'stack'
        self.block = block
        self.dest = dest

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.block, self.dest)

    def __str__(self):
        return 'Stack %s onto %s' % (self.block, self.dest)

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return other.type == self.type \
            and other.block == self.block \
            and other.dest == self.dest

    def __hash__(self):
        return hash(self.type) ^ hash(self.block) ^ hash(self.dest)


def state_picture(below_spec):
    """Draw some towers of blocks"""
    blocks = set(range(1, len(below_spec) + 1))
    above_me = {
        bot: top
        for top, bot in enumerate(below_spec, start=1) if bot != 0
    }
    blocks_on_table = {b for b in blocks if below_spec[b - 1] == 0}
    clear_blocks = {b for b in blocks if above_me.get(b) is None}
    # create towers; this sorts by base of tower and ensures that blank spaces
    # are left for blocks not on the table, thus ensuring a consistent
    # left-right tower arrangement
    towers = []
    for block in sorted(blocks):
        if block in blocks_on_table:
            tower = [block]
            while tower[-1] not in clear_blocks:
                above = above_me[tower[-1]]
                tower.append(above)
            towers.append(tower)
        else:
            # empty tower
            towers.append([])
    max_height = max(map(len, towers))
    tower_width = max(len(str(b)) for b in blocks)
    rows = []
    for height in range(max_height - 1, -1, -1):
        row = []
        for tower in towers:
            if len(tower) > height:
                this_block = str(tower[height])
                row.append(this_block.rjust(tower_width, ' '))
            else:
                row.append(' ' * tower_width)
        rows.append(' '.join(row))
    rows.append('-' * ((tower_width + 1) * len(towers) - 1))
    tower_pic = '\n'.join(rows)
    return tower_pic


def cgi_make_action(act_str):
    """Turn an action string (e.g. 'X to table', 'X to Y') into an actual
    Action data structure."""
    if ':' in act_str:
        desc = act_str.split(':', 1)[1].strip()
    else:
        desc = act_str.strip()
    spl = desc.split()
    assert 'to' in spl, "desc '%s' invalid" % desc
    if desc.endswith('to table'):
        return Unstack(int(spl[0]))
    return Stack(int(spl[0]), int(spl[-1]))


def manual_make_actions(act_strs, sorted_block_names):
    # This parses a plan of the form ['unstack b1', 'stack b1 b2', …etc].
    # Action names:
    #
    #  - pick-up(?block) (pick up ?block from table with gripper)
    #  - unstack(?block, ?base) (pick up from another block with
    #    gripper)
    #  - put-down(?block) (release ?block from gripper onto table)
    #  - stack(?block, ?target) (release block ?block onto other block ?target)
    #
    # Of course a pick-up or unstack action must be followed by a put-down or
    # stack action, so we can still pair these into normal actions.
    name_to_id = {
        block_name: block_id
        for block_id, block_name in enumerate(sorted_block_names, start=1)
    }
    n_acts = len(act_strs)
    even_num = (n_acts % 2) == 0
    if not even_num:
        if n_acts < 299:
            raise ValueError(
                "expected even number of actions, but got %d actions" % n_acts)
        warnings.warn("Got odd # of actions (%d); will truncate under "
                      "assumption that it's a long failed plan" % n_acts)
        act_strs = act_strs[:-1]
    actions = []
    held_block = None
    try:
        for act_str in act_strs:
            parts = act_str.strip(string.whitespace + '()').split()
            n_parts = len(parts)
            if parts[0] == 'pick-up':
                # handle pick-up(?block)
                assert n_parts == 2, "wrong token count"
                assert held_block is None, "already holding block"
                assert parts[1] in name_to_id, "undeclared block"
                held_block = name_to_id[parts[1]]
            elif parts[0] == 'unstack':
                # handle unstack(?block, ?base)
                assert n_parts == 3, "wrong token count"
                assert held_block is None, "already holding block"
                assert parts[1] in name_to_id \
                    and parts[2] in name_to_id, "undeclared block(s)"
                held_block = name_to_id[parts[1]]
            elif parts[0] == 'put-down':
                # handle put-down(?block)
                assert n_parts == 2, "wrong token count"
                assert parts[1] in name_to_id, "undeclared block"
                assert held_block == name_to_id[parts[1]], \
                    "holding wrong block"
                actions.append(Unstack(held_block))
                held_block = None
            elif parts[0] == 'stack':
                # handle stack(?block, ?target)
                assert n_parts == 3, "wrong token count"
                assert parts[1] in name_to_id \
                    and parts[2] in name_to_id, "undeclared block(s)"
                assert held_block == name_to_id[parts[1]], \
                    "holding wrong block"
                target_block = name_to_id[parts[2]]
                actions.append(Stack(held_block, target_block))
                held_block = None
            else:
                assert False, "unrecognised part"
    except AssertionError as e:
        if len(e.args) == 1:
            message, = e.args
        else:
            message = 'no message'
        raise SyntaxError("Could not parse action string '%s' (%s)" %
                          (act_str, message))
    return actions


def parse_manual_plan(manual_plan_path, sorted_block_names):
    with open(manual_plan_path, 'r') as plan_fp:
        if os.path.basename(manual_plan_path) == 'results.json':
            json_data = json.load(plan_fp)
            # grab first trajectory and cut off the "GOAL! :D" or "FAIL! D:" at
            # the end of the trajectory
            act_lines = json_data['trial_paths'][0][:-1]
        else:
            stripped_lines = (l.strip() for l in plan_fp)
            act_lines = [l for l in stripped_lines if l]
    actions = manual_make_actions(act_lines, sorted_block_names)
    return actions


# rename common predicates to a single canonical form
CANONICAL_PRED_NAMES = {'handempty': 'emptyhand', 'ontable': 'on-table'}


def props_to_below_dict(props):
    """Maps propositions (each prop is list of strings) to dict d saying "d[b]
    is below b" for each block b."""
    block_below = {}
    for prop in props:
        prop[0] = CANONICAL_PRED_NAMES.get(prop[0], prop[0])
        if prop[0] == 'on':
            top, bot = prop[1:]
            block_below[top] = bot
        elif prop[0] == 'on-table':
            block = prop[1]
            block_below[block] = None
        else:
            # we don't handle these, and don't care
            if prop[0] not in {'emptyhand', 'clear'}:
                raise ValueError("Unrecognised proposition %s" % (prop[0], ))
    return block_below


def block_sort_key(block_name):
    number_strs = re.findall(r'\d+', block_name)
    numbers = tuple(map(int, number_strs))
    # sort first by numbers, then by block name
    return (numbers, block_name)


def parse_file(ppddl_txt):
    """Parse a PPDDL file to retrieve initial state, goal state, and list of
    blocks."""
    # the [0] gets rid of the file-wide pseudo-node (below which all (define)s,
    # (problem)s, etc. live).
    root = parse_sexprs(ppddl_txt)[0]

    # make a list of things in each section
    sect_dict = {}
    for child in root:
        if isinstance(child, HList):
            sect_dict[child[0]] = child[1:]

    # get block names, ignoring '- block' type specifier at end
    obj_sect = sect_dict[':objects']
    assert obj_sect[-2:] == ['-', 'block'], obj_sect
    sorted_block_names = sorted(obj_sect[:-2], key=block_sort_key)

    # now get init state
    init_below = props_to_below_dict(sect_dict[':init'])
    # check block names valid
    set_block_names = set(sorted_block_names)
    assert all(b in set_block_names for b in init_below.keys())

    # get goal state
    # [0][1:] to get rid of the (and ...)
    goal_below = props_to_below_dict(sect_dict[':goal'][0][1:])
    # another name validity check
    assert all(b in set_block_names for b in goal_below.keys())

    return init_below, goal_below, sorted_block_names


def below_to_spec(block_below, sorted_block_names):
    """Turns dict produced by props_to_below_dict into integer array expected
    by BW solver."""
    block_ids = {
        tower_name: idx
        # number things from 1 onwards
        for idx, tower_name in enumerate(sorted_block_names, start=1)
    }
    int_list = []
    for block_name in sorted_block_names:
        below_name = block_below[block_name]
        if below_name is None:
            # table
            below_id = 0
        else:
            # other block
            below_id = block_ids[below_name]
        int_list.append(below_id)
    return int_list


def _retry_post(*args, retries=10, **kwargs):
    """Retries requests.post with given args/kwargs for at most `retries` times
    (or until success)."""
    assert retries > 0
    for retry in range(1, retries + 1):
        try:
            return requests.post(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            if retry < retries:
                continue
            raise


def cgi_solution_plan(start_spec, goal_spec, plan_type):
    """Run John Slaney's CGI script to get a plan."""
    assert len(start_spec) == len(goal_spec)
    assert all(isinstance(s, int) for s in start_spec)
    assert all(isinstance(s, int) for s in goal_spec)
    form_data = {
        'verb': SCRIPT_VERB,
        'mode': SCRIPT_MODE,
        'initial': ' '.join(map(str, start_spec)),
        'goal': ' '.join(map(str, goal_spec)),
        'size': len(start_spec),
        'algo': int(plan_type)
    }
    resp = _retry_post(
        'http://users.cecs.anu.edu.au/~jks/cgi-bin/bwstates/bwoptcgi',
        data=form_data,
        # sometimes John's CGI script can be slow
        timeout=120)
    lines = resp.text.splitlines()
    last_lines = lines[lines.index('Plan:') + 1:]
    plan_lines = [l.strip() for l in last_lines if l.strip()]
    return list(map(cgi_make_action, plan_lines))


def expected_pol_cost(det_plan, is_det):
    """Get expected cost of (possibly probabilistic) blocks world policy
    corresponding to given deterministic blocks world plan."""
    cost = 0
    for action in det_plan:
        if action.type == 'unstack':
            # in BW each unstack requires two actions (grab & deposit);
            # in PBW each unstack takes 1.75x expected cost
            cost += 2 if is_det else 1.75
        else:
            assert action.type == 'stack'
            # in BW each stack also requires two actions (grab & deposit);
            # in PBW each stack takes 28/9x (~3.11x) expected cost
            cost += 2 if is_det else 28.0 / 9
    return cost


class InvalidAction(Exception):
    def __init__(self, spec, action, msg):
        self.spec = spec
        self.action = action
        self.msg = msg


def apply_action(old_spec, action):
    # figure out which blocks can be grabbed or used as bases, for
    # sanity-checking purposes
    clear_blocks = sorted(set(range(1, len(old_spec) + 1)) - set(old_spec))

    # this is the new `below` dict that we'll return
    new_spec = list(old_spec)

    if isinstance(action, Unstack):
        # make sure it's not below anything
        if action.block not in clear_blocks:
            raise InvalidAction(
                old_spec, action,
                "block '%s' not clear so cannot be unstacked" %
                (action.block, ))
        new_spec[action.block - 1] = 0
    elif isinstance(action, Stack):
        # nothing can be above action.dest, or above action.block, or below
        # action.block
        if action.block not in clear_blocks:
            raise InvalidAction(
                old_spec, action,
                "block '%s' not clear so cannot be stacked on anything" %
                (action.block, ))
        # "noop" stacks (where we unstack a block & return it to its previous
        # position) are permissible. In the underlying plan, such a situation
        # corresponds to a sequence of moves like "(unstack a b) (put-down a)
        # (pick-up a) (stack a b)".
        is_noop = old_spec[action.block - 1] == action.dest
        if action.dest not in clear_blocks and not is_noop:
            raise InvalidAction(
                old_spec, action,
                "block '%s' not clear so cannot be stacked onto" %
                (action.dest, ))
        new_spec[action.block - 1] = action.dest
    else:
        raise TypeError("can't deal with action %r" % (action, ))

    return new_spec


def verify_det_plan(plan, init_spec, goal_spec):
    current_spec = init_spec
    plan = list(plan)
    n_actions = len(plan)
    for action_num, action in enumerate(plan):
        try:
            current_spec = apply_action(current_spec, action)
        except InvalidAction as inval_ex:
            inval_msg = inval_ex.msg + " (step %d/%d)" % (action_num,
                                                          n_actions)
            raise ValueError(inval_msg)
    # make sure that final state we end up in is equal to the actual goal state
    if goal_spec is not None:
        return current_spec == goal_spec


def print_det_plan(plan, init_spec, goal_spec):
    current_spec = init_spec
    print('Goal state:')
    print(state_picture(goal_spec))
    print('\nInitial state:')
    print(state_picture(current_spec))
    for action in plan:
        print('\n' + str(action))
        current_spec = apply_action(current_spec, action)
        print(state_picture(current_spec))


parser = argparse.ArgumentParser(
    description='try solving PBW instances or something')
parser.add_argument('--strategy',
                    default=[],
                    choices=VALID_STRATS,
                    action='append',
                    help='planner strategy (can repeat)')
parser.add_argument(
    '--manual-plan',
    default=None,
    help='path to .txt file containing actions in manual plan (one per line, '
    'parens optional); useful when using `--strategy MANUAL`')
parser.add_argument(
    '--pretty-mode',
    choices=('off', 'init-goal', 'full-plan'),
    default='off',
    dest='draw_pic_mode',
    help='determines whether to draw entire plan (full-plan), initial state '
    'and goal only (init-goal), or nothing (off; the default)')
parser.add_argument(
    '--trim-cycles',
    default=False,
    action='store_true',
    help='trim cycles from the end of plans (useful for manual plans)')
parser.add_argument('--allow-invalid-plans',
                    action='store_true',
                    default=False,
                    help='silently ignore plans that do not reach goal')
parser.add_argument(
    'domain_type',
    choices=('prob', 'det'),
    help='type of BW domain to solve (either probabilistic or deterministic)')
parser.add_argument('ppddl_file',
                    nargs='+',
                    type=argparse.FileType('r'),
                    help='file to read (can repeat)')


def main():
    args = parser.parse_args()
    for ppddl_file in args.ppddl_file:
        base_fn = os.path.basename(ppddl_file.name)
        ppddl_txt = ppddl_file.read()
        init_below, goal_below, sorted_blocks = parse_file(ppddl_txt)
        init = below_to_spec(init_below, sorted_blocks)
        goal = below_to_spec(goal_below, sorted_blocks)
        if args.draw_pic_mode == 'init-goal':
            print('Initial state:')
            print(state_picture(init))
            print('')
            print('Goal state:')
            print(state_picture(goal))
        for strat_str in args.strategy:
            strat = getattr(SolutionType, strat_str)
            if strat == SolutionType.MANUAL:
                assert args.manual_plan is not None, \
                    "`--strategy MANUAL` requires a `--manual-plan " \
                    "<file>` argument"
                det_plan = parse_manual_plan(args.manual_plan, sorted_blocks)
                if args.trim_cycles:
                    det_plan, _ = remove_cycles(det_plan)
            else:
                assert not args.trim_cycles, \
                    "--trim-cycles only makes sense for manual plans; " \
                    "plans from the solvers don't have cycles"
                det_plan = cgi_solution_plan(init, goal, strat)
            is_valid_plan = verify_det_plan(det_plan, init, goal)
            if not is_valid_plan:
                msg = "%d-step %s plan for '%s' does not reach goal!" % \
                    (len(det_plan), strat_str, ppddl_file.name)
                if args.allow_invalid_plans:
                    # warn, but okay
                    warnings.warn(msg)
                else:
                    # error!
                    raise Exception(msg)
            pol_cost = expected_pol_cost(det_plan,
                                         is_det=args.domain_type == 'det')
            print('RESULT|File {base_fn}|Method {strat_str}|Cost '
                  '{pol_cost}|Actions {n_actions}|Valid '
                  '{is_valid_plan}'.format(
                      base_fn=base_fn,
                      strat_str=strat_str,
                      pol_cost=pol_cost,
                      n_actions=len(det_plan),
                      is_valid_plan=is_valid_plan))
            if args.draw_pic_mode == 'full-plan':
                print('START PLAN VIS')
                print_det_plan(det_plan, init, goal)
                print('END PLAN VIS')


if __name__ == '__main__':
    main()
