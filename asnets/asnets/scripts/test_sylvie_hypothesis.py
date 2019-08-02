#!/usr/bin/env python3
"""Test Sylvie's hypothesis that ASNets are doing US with some simple "skip the
table when you can make a constructive move" optimisation."""

import argparse

from tqdm import tqdm

from bw_collate_vis import discover_result_jsons, index_pddls
from pbw_solve import manual_make_actions, parse_file, below_to_spec, \
    apply_action


def has_constructive_move(block, spec, goal_spec):
    nblocks = len(spec)
    free_blocks = set(range(1, nblocks + 1)) - set(spec)
    if goal_spec[block - 1] == 0:
        # target is table, just check that it's free & not already on table
        return block in free_blocks and not spec[block - 1] == 0
    else:
        # otherwise make sure that target block is in position
        if not is_in_position(goal_spec[block - 1], spec, goal_spec):
            return False
        # also make sure that this block is *not* in position
        if is_in_position(block, spec, goal_spec):
            return False
        # make sure that this block and the target are both free
        return {block, goal_spec[block - 1]} <= free_blocks


def is_in_position(block, spec, goal_spec):
    # make sure that entire tower for this block is sensible
    while block != 0:
        if spec[block - 1] != goal_spec[block - 1]:
            return False
        block = goal_spec[block - 1]
    return True


def verify_hypothesis(pddl_path, tentative_plan_strs):
    with open(pddl_path, 'r') as pddl_fp:
        init_below, goal_below, sorted_blocks \
            = parse_file(pddl_fp.read())
    actions = manual_make_actions(tentative_plan_strs, sorted_blocks)
    goal_spec = below_to_spec(goal_below, sorted_blocks)
    current_spec = below_to_spec(init_below, sorted_blocks)
    nactions = len(actions)
    reasons = []
    hypothesis_satisfied = True
    for action_num, action in enumerate(actions, start=1):
        if action.type == 'unstack':
            # move to table; need to check that either block is the base of a
            # tower, or that no constructive move exists block exists
            is_constructive = goal_spec[action.block - 1] == 0
            could_have_been_constructive = has_constructive_move(
                action.block, current_spec, goal_spec)
            if not is_constructive and could_have_been_constructive:
                hypothesis_satisfied = False
                reason = "non-constructive move %s (move %d/%d) taken " \
                    "in lieu of an available constructive move" \
                    % (action, action_num, nactions)
                reasons.append(reason)
            current_spec = apply_action(current_spec, action)
        else:
            # move from block to block; need to check that this is a
            # constructive move
            assert action.type == 'stack'
            new_spec = apply_action(current_spec, action)
            if not is_in_position(action.block, new_spec, goal_spec):
                if has_constructive_move(action.block, current_spec,
                                         goal_spec):
                    reason = "%s (move %d/%d) is not constructive, and " \
                        "constructive move *was* available!" \
                        % (action, action_num, nactions)
                    hypothesis_satisfied = False
                    reasons.append(reason)
            current_spec = new_spec
    return hypothesis_satisfied, reasons


parser = argparse.ArgumentParser(
    description="test Sylvie's hypothesis that ASNets are doing US with "
    "simple block-to-block optimisation (if on(X,Y) in goal, and Y & X are "
    "clear, then stick X straight on Y)")
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

    exceptions = []
    nproblems = 0
    for result_json_path, result_data in tqdm(result_jsons):
        if not result_data["no_train"]:
            # skip training instances since IIRC traces are produced
            # stochastically & thus may not reflect actual policy
            continue
        if not all(result_data['all_goal_reached']):
            print('WARNING: fail plan at %s' % result_json_path)
            continue

        asnet_plan = result_data['trial_paths'][0][:-1]
        prob_name = result_data['problem']
        pddl_path = pddl_index[prob_name]
        satisfied, reasons = verify_hypothesis(pddl_path, asnet_plan)
        if not satisfied:
            exceptions.append((result_json_path, pddl_path, reasons))
        nproblems += 1
    nexceptions = len(exceptions)
    print('Found %d exceptions from %d problems (%.g%%)' %
          (nexceptions, nproblems, 100.0 * nexceptions / nproblems))
    print('Overview of exceptions')
    for result_json_path, pddl_path, reasons in exceptions:
        print('For %s, %s (%d):' % (result_json_path, pddl_path, len(reasons)))
        for reason in reasons:
            print('  %s' % reason)


if __name__ == '__main__':
    main()
