"""Visualising gold-miner problems & plans."""

import argparse
import collections
import datetime
import enum
import io
import os
import os.path as osp
import re

import crayons
import numpy as np
import rpyc
import tqdm

from asnets.multiprob import ProblemServer
from asnets.prof_utils import can_profile
from asnets.py_utils import remove_cycles
from asnets.scripts.run_asnets import get_problem_names
from asnets.ssipp_interface import Cutter
from asnets.state_reprs import sample_next_state, get_init_cstate
from asnets.supervised import ProblemServiceConfig, PlannerExtensions

LOC_RE = re.compile(r'^f(\d+)-(\d+)f$')

DARK_BROWN = [139 / 255.0, 69 / 255.0, 19 / 255.0]
LIGHT_BROWN = [205 / 255.0, 133 / 255.0, 63 / 255.0]
GREY = [190 / 255.0, 190 / 255.0, 190 / 255.0]


class CellBase(enum.Enum):
    CLEAR = 'clear'
    HARD_ROCK = 'hard-rock-at'
    SOFT_ROCK = 'soft-rock-at'


BASE_BY_PRED_NAME = {enum_val.value: enum_val for enum_val in CellBase}
COLOUR_BY_ENUM = {
    CellBase.CLEAR: GREY,
    CellBase.HARD_ROCK: DARK_BROWN,
    CellBase.SOFT_ROCK: LIGHT_BROWN,
}


def _parse_loc(loc):
    # parse location of the form 'fX-Yf' (e.g 'f4-1f')
    # (if I ever use a non-default generator then I'll need smarter parsing
    # code that doesn't rely on names)
    (xs, ys), = LOC_RE.findall(loc)
    loc_tup = (int(xs), int(ys))
    return loc_tup


def _unparse_loc(loc):
    x, y = loc
    return 'f%d-%df' % loc


class GMState(object):
    """Class to represent & do computations on states from GoldMiner. Only
    works on the very simple square grids produced by the IPC 2008 generator
    (it will error out if you have more than one laser location, for
    example)."""

    def __init__(self, cstate):
        prop_truth_mask = cstate.props_true
        props_by_pred = {}
        for bprop, truth in prop_truth_mask:
            props_by_pred.setdefault(bprop.pred_name, []) \
                .append((bprop, truth))
        n_connected = len(props_by_pred['connected'])
        # connections are bidirectional, and only connect interior things
        self.size = int((1 + np.sqrt(1 + n_connected)) / 2)
        # sanity check
        assert n_connected == 4 * self.size * (self.size - 1)

        # now make underlying grid showing whether each location is soft
        # rock/hard rock/clear space
        self.grid_contents = np.ndarray((self.size, self.size), dtype='object')
        for bprop, truth in prop_truth_mask:
            if bprop.pred_name not in BASE_BY_PRED_NAME or not truth:
                continue
            x, y = _parse_loc(*bprop.arguments)
            new_base = BASE_BY_PRED_NAME[bprop.pred_name]
            assert self.grid_contents[x, y] is None, \
                "conflict at (x=%d, y=%d): %r (old) vs %r (new)" \
                % (x, y, self.grid_contents[x, y], new_base)
            self.grid_contents[x, y] = new_base

        # make sure everything has a type
        assert not np.any(self.grid_contents == None), \
            "can't have any empty cells in grid: %s" \
            % self.grid_contents  # noqa: E711

        # make sure the top row is clear (like it is in generated instances)
        assert np.all(self.grid_contents[:, 0] == CellBase.CLEAR), \
            "top row should be clear of obstructions"

        # things that may be located somewhere in the grid
        self.bomb_loc = None
        self.gold_loc = None
        self.laser_loc = None
        self.robot_loc = None
        # things that we may be holding
        self.has_bomb = None
        self.has_gold = None
        self.has_laser = None

        obj_types = ['bomb', 'gold', 'laser', 'robot']
        for loc_obj_type in obj_types:
            pred_name = loc_obj_type + '-at'
            locs = [
                _parse_loc(*bprop.arguments)
                for bprop, truth in props_by_pred[pred_name] if truth
            ]
            if len(locs) == 0:
                # nothing to do
                continue
            assert len(locs) == 1
            (x, y), = locs
            attr_name = loc_obj_type + '_loc'
            new_val = (x, y)
            assert getattr(self, attr_name) is None, \
                "conflicting values for %s: %r vs %r" % \
                (attr_name, getattr(self, attr_name), new_val)
            setattr(self, attr_name, new_val)

        has_types = ['bomb', 'gold', 'laser']
        for has_obj_type in has_types:
            (_, truth), = props_by_pred['holds-' + has_obj_type]
            attr_name = 'has_' + has_obj_type
            assert getattr(self, attr_name) is None, \
                "conflicting values for %s: %r vs %r" % \
                (attr_name, getattr(self, attr_name), new_val)
            setattr(self, attr_name, truth)

        # time for some more sanity checks! yay!
        # first, there should always be a location for robot and bomb
        assert self.bomb_loc is not None
        assert self.robot_loc is not None

        # also we should always know whether or not we have laser/gold
        assert self.has_bomb is not None
        assert self.has_gold is not None
        assert self.has_laser is not None

        # if the laser isn't at any location, then we should be holding it! the
        # same does not go for gold, which can be destroyed
        assert self.has_laser or self.laser_loc is not None

        # make sure that gold is at the bottom
        if self.gold_loc:
            assert self.gold_loc[1] == self.size - 1

        # make sure that the robot's cell is clear
        assert self.grid_contents[self.robot_loc] == CellBase.CLEAR

        # cached values that might get filled in later via flood fill
        self._dist_grid = self._parent_grid = None

    def make_rgb_grid(self):
        """Make RGB underlay of rock/soft rock/clear cells for grid
        visualisation."""
        grid_rgb = np.zeros((self.size, self.size, 3))
        for y in range(self.size):
            for x in range(self.size):
                base_val = self.grid_contents[x, y]
                new_colour = COLOUR_BY_ENUM[base_val]
                # Note the sneaky transpose b/c this is an image with (y,x)
                # (i.e row-wise) indexing. Honestly I'm not sure why I didn't
                # just rename them "row" and "column" instead of "x" and "y"
                # so I could use consistent naming throughout, but I guess that
                # what's done is done.
                grid_rgb[y, x] = new_colour
        return grid_rgb

    def robot_flood_fill(self):
        """Flood fill all clear squares of grid from robot's location out.
        Useful for figuring out which direction the robot should go in order to
        get to some reachable location. Given that there are O(self.size) clear
        squares in most GM plans, this should usually be about as efficient as
        A* or whatever (in terms of states explored)."""

        if self._dist_grid is not None:
            assert self._parent_grid is not None
            return self._dist_grid, self._parent_grid

        clear_grid = self.grid_contents == CellBase.CLEAR
        to_expand = collections.deque()
        to_expand.append(self.robot_loc)

        # grids are indexed (x,y), unlike images
        dist_grid = np.full((self.size, self.size),
                            float('inf'),
                            dtype='float')
        dist_grid[self.robot_loc] = 0

        # also indexed (x,y) and stores (x,y) coordinates
        parent_grid = np.zeros((self.size, self.size), dtype='object')
        while len(to_expand) > 0:
            x, y = to_expand.popleft()
            dist = dist_grid[x, y]
            assert np.isfinite(dist)

            neighs = [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
            ]
            for dx, dy in neighs:
                xp = x + dx
                yp = y + dy
                if xp < 0 or xp >= self.size or yp < 0 or yp >= self.size:
                    continue

                old_dist = dist_grid[xp, yp]

                if clear_grid[xp, yp] and not np.isfinite(old_dist):
                    dist_grid[xp, yp] = dist + 1
                    assert np.isfinite(dist_grid[xp, yp])
                    parent_grid[xp, yp] = (x, y)
                    to_expand.append((xp, yp))

        self._dist_grid = dist_grid
        self._parent_grid = parent_grid

        return dist_grid, parent_grid


def render_gm_state(cstate):
    """Render a GoldMiner state---as represented by a CState---to an RGB
    array."""
    import matplotlib.pyplot as plt
    import matplotlib.offsetbox as obox
    import matplotlib.ticker as tick

    # some housekeeping
    gm_state = GMState(cstate)

    fig = plt.figure()
    ax = plt.gca()

    # now make underlying grid showing whether each location is soft rock/hard
    # rock/clear space
    grid_rgb = gm_state.make_rgb_grid()
    plt.imshow(grid_rgb)

    asset_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'assets')
    obj_types = ['bomb', 'gold', 'laser', 'robot']
    for obj_type in obj_types:
        sprite_path = osp.join(asset_dir, obj_type + '.png')
        image = obox.OffsetImage(plt.imread(sprite_path), zoom=1)
        location = getattr(gm_state, obj_type + '_loc')
        if location is not None:
            x, y = location
            artist = obox.AnnotationBbox(image, (x, y), frameon=False)
            ax.add_artist(artist)

    # finally, draw whatever the robot has in its little robot hands (bomb,
    # gold, laser, etc.)
    hold_obj_types = ['bomb', 'gold', 'laser']
    for obj_type in hold_obj_types:
        sprite_path = osp.join(asset_dir, obj_type + '.png')
        image = obox.OffsetImage(plt.imread(sprite_path), zoom=0.5)
        holding_thing = getattr(gm_state, 'has_' + obj_type)
        if not holding_thing:
            continue
        xoff, yoff = gm_state.robot_loc
        xoff += 0.3
        yoff += 0.3
        # does this work? Who knows?
        artist = obox.AnnotationBbox(image, (xoff, yoff), frameon=False)
        ax.add_artist(artist)

    # the nuclear option for getting rid of blank space, thanks to
    # https://stackoverflow.com/a/27227718
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(tick.NullLocator())
    ax.yaxis.set_major_locator(tick.NullLocator())

    # render to array
    out_buffer = io.BytesIO()
    fig.savefig(out_buffer, bbox_inches='tight', pad_inches=0)
    out_buffer.seek(0)
    data = plt.imread(out_buffer)

    plt.close(fig)

    return data


def parse_gm_plan(plan_txt):
    # first construct regex that finds action name followed by zero or more
    # location identifiers (greedy matching, so it should get all of them)
    actions = [
        'move', 'pickup-laser', 'pickup-bomb', 'putdown-laser',
        'detonate-bomb', 'fire-laser', 'pick-gold'
    ]
    action_re = r'(%s)' % '|'.join(actions)
    loc_re = r'f(\d+)-(\d+)f'
    full_re = re.compile(r'(%s(\s+%s)*)' % (action_re, loc_re))

    # now apply regex to possible plan text file & return all matching actions
    action_groups = full_re.findall(plan_txt)
    actions = [head for head, *_ in action_groups]

    return actions


def parse_gm_plan_file(plan_path):
    with open(plan_path, 'r') as plan_fp:
        actions = parse_gm_plan(plan_fp.read())
    if not actions:
        raise ValueError("No actions found in '%s'. Is it a legit plan?" %
                         plan_path)
    return actions


@can_profile
def gold_miner_planner(cstate):
    gm_state = GMState(cstate)
    if gm_state.has_gold:
        # we're done! this should probably not be happening, but we'll only
        # return empty plan to signal our displeasure >:(
        # (passive-aggressive planning!)
        assert cstate.is_goal, \
            "if we have the gold then it better be a goal state :P"
        return []

    if gm_state.gold_loc is None and not gm_state.has_gold:
        # gold has been destroyed by laser :'(
        # (this is one of two kinds of avoidable dead ends I know of in
        # GM---the other is when you pick up a bomb, but there is no exposed
        # soft rock to blast it with!)
        return None

    # so, three phases:
    #
    # (1) >1 cells left before gold, so we need to get laser & blow the cells
    #     up until we're close to the gold.
    # (2) One cell left before gold, so we need to get bomb and blow it up (but
    #     softly, because that's what bombs do apparently).
    # (3) Way to gold is clear & we can go get it!
    #
    # Really I should check conditions in reverse order.
    dist_grid, parent_loc_grid = gm_state.robot_flood_fill()

    def gen_move_actions(target_loc):
        """Backtrack in order to find list of actions taking the robot from its
        current location to target_loc. target_loc must be reachable!"""
        assert np.isfinite(dist_grid[target_loc]), \
            "asked for path to unreachable point %s (from robot loc %s)" % \
            (target_loc, gm_state.robot_loc)
        actions_rev = []
        current_loc = target_loc
        while current_loc != gm_state.robot_loc:
            assert len(actions_rev) < 1000, \
                "uh this looks like an infinite loop, maybe you should fix?"
            last_loc = parent_loc_grid[current_loc]
            # this is the unique ident of the action; we'll have to figure out
            # the index some other way
            move_name = 'move %s %s' % (_unparse_loc(last_loc),
                                        _unparse_loc(current_loc))
            actions_rev.append(move_name)
            current_loc = last_loc
        actions = actions_rev[::-1]
        return actions

    pickup_gold_action = 'pick-gold %s' % _unparse_loc(gm_state.gold_loc)
    if np.isfinite(dist_grid[gm_state.gold_loc]):
        # We've got a clear path! Take it!
        actions = gen_move_actions(gm_state.gold_loc)
        actions.append(pickup_gold_action)
        return actions

    # if gold location has rock, but one of its neighbours does not (and is
    # reachable), then it's BOMBING TIME!
    gx, gy = gm_state.gold_loc
    neighbours = [
        # (dx+dx, gy+dy)
        (gx, gy - 1),
        (gx - 1, gy),
        (gx + 1, gy),
        (gx, gy + 1),
    ]
    neighbour_dists = [
        (dist_grid[n], n) for n in neighbours
        if 0 <= n[0] < gm_state.size and 0 <= n[1] < gm_state.size
    ]
    best_dist, best_neigh = min(neighbour_dists)
    gold_is_rocked \
        = gm_state.grid_contents[gm_state.gold_loc] != CellBase.CLEAR
    if np.isfinite(best_dist) and gold_is_rocked:
        # bomb path: drop laser if we have it, get bomb, blow up gold rock
        actions = []
        if gm_state.has_laser:
            # drop laser where we are
            actions.append('putdown-laser %s' %
                           _unparse_loc(gm_state.robot_loc))
        if not gm_state.has_bomb:
            # go to bomb, pick it up; we'll blow things up next time
            actions.extend(gen_move_actions(gm_state.bomb_loc))
            actions.append('pickup-bomb ' + _unparse_loc(gm_state.bomb_loc))
            return actions
        # otherwise, go to the best neighbour
        actions.extend(gen_move_actions(best_neigh))
        # blow up gold loc so we can get gold next time
        actions.append(
            'detonate-bomb %s %s' %
            (_unparse_loc(best_neigh), _unparse_loc(gm_state.gold_loc)))
        return actions

    # If we get down to here, then there's at least one unreachable square
    # adjacent to the rock thing, so it's LASER TIME! Plan consists of several
    # stages:
    # (1) Error out if you have a bomb. Ideally, you'd instead dispose of the
    #     bomb by finding reachable soft rock nearest the gold & blowing it up,
    #     but that seems like too much effort for a strategy that the ASNet
    #     probably can't copy anyway.
    # (2) go pick up laser, if we haven't already, and error out if we can't
    # (3) move to lowest reachable grid cell above gold
    # (4) blast straight down until we're within one cell of the gold (i.e
    #     until the bomb path condition is satisfied)

    if gm_state.has_bomb:
        # TODO: fix this so that it actually disposes of the bomb properly
        # instead of erroring out
        return None

    if not gm_state.has_laser:
        return [
            *gen_move_actions(gm_state.laser_loc),
            'pickup-laser ' + _unparse_loc(gm_state.laser_loc)
        ]

    best_y = 0
    for check_y in range(0, gy - 1):
        # walk down the column until we hit first rock
        if not np.isfinite(dist_grid[gx, check_y]):
            break
        best_y = check_y
    target_clear = (gx, best_y)
    target_rock = (gx, best_y + 1)
    # ah better make sure it is a rock that we hit, too
    assert gm_state.grid_contents[target_clear] == CellBase.CLEAR
    assert gm_state.grid_contents[target_rock] != CellBase.CLEAR
    # TODO: I should produce plan to blast out the column of rock all in one go
    return [
        *gen_move_actions(target_clear),
        # blast away!
        'fire-laser %s %s' %
        (_unparse_loc(target_clear), _unparse_loc(target_rock))
    ]


# #################################### #
# DEMO CODE TO SHOW OFF VISUALISATIONS #
# #################################### #


def demo():
    # get arg path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-dir',
        default='gm-plan-out',
        help='output dir to write plan frames to (if --plan given)')
    parser.add_argument(
        '--max-cycle-repeat',
        default=4,
        help='max num cycles to show for cyclic plan tails (0 for unlimited)')
    parser.add_argument('--plan',
                        default=None,
                        help='path to .txt file for plan')
    parser.add_argument('domain_pddl', help='path to PDDL domain')
    parser.add_argument('problem_pddl', help='path to PDDL problem')
    args = parser.parse_args()

    print("Setting up MDPSim/SSiPP crap")
    rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
    all_pddl = [args.domain_pddl, args.problem_pddl]
    prob_name = get_problem_names(all_pddl)[0]
    service_config = ProblemServiceConfig(all_pddl,
                                          prob_name,
                                          teacher_heur='h-add',
                                          use_lm_cuts=True,
                                          teacher_planner='fd')
    problem_server = ProblemServer(service_config)
    problem_server.service.initialise()
    # single_problem = SingleProblem(prob_name, problem_server)
    planner_exts = PlannerExtensions(service_config.pddl_files,
                                     service_config.init_problem_name,
                                     dg_use_lm_cuts=True,
                                     dg_use_act_history=True)

    # split rest of this out into separate functions so that I don't pollute
    # their scopes (& also so that flake8 provides better warnings)
    if not args.plan:
        main_interactive(args, planner_exts)
    else:
        main_plan(args, planner_exts)


def get_action_freqs(state,
                     planner_exts,
                     *,
                     include_zero=False,
                     include_disabled=True):
    # maps action name to # of times it has been executed
    extra_dim = max(state._aux_data_interp_to_id.values()) + 1
    aux_reshaped = state.aux_data.reshape((-1, extra_dim))
    count_dim = state._aux_data_interp_to_id['action_count']
    count_vec = aux_reshaped[:, count_dim]
    assert len(count_vec) == len(state.acts_enabled)
    action_freqs = []
    for (bact, enabled), count in zip(state.acts_enabled, count_vec):
        if count == 0 and not include_zero:
            continue
        if not (enabled or include_disabled):
            continue
        action_freqs.append((count, bact.unique_ident))
    return sorted(action_freqs)


def main_interactive(args, planner_exts):
    import matplotlib.pyplot as plt

    print("Time to play a game!")
    state = get_init_cstate(planner_exts)
    shown = False
    from asnets.teacher import DomainSpecificTeacher
    ds_teacher = DomainSpecificTeacher(planner_exts)
    while True:
        cutter = Cutter(planner_exts)
        action_cuts = cutter.get_action_cuts(state.to_ssipp(planner_exts))
        enabled = [a for a, m in state.acts_enabled if m]
        ssipp_enabled = {'(%s)' % a.unique_ident for a in enabled}
        print('Action cuts (%d):' % len(action_cuts))
        greens = set()
        for action_cut in action_cuts:
            print(' ', end='')
            for cut_elem in action_cut:
                to_print = ' %s' % cut_elem
                if cut_elem in ssipp_enabled:
                    greens.add(cut_elem[1:-1])
                    to_print = crayons.green(to_print)
                print(to_print, end='')
            print()
        print("Action frequencies: ")
        action_freqs = get_action_freqs(state, planner_exts)
        for act_count, act_name in action_freqs:
            print('  %s (%d)' % (act_name, act_count))
        gm_image = render_gm_state(state)
        if not shown:
            plt.ion()
            im_handle = plt.imshow(gm_image)
        else:
            im_handle.set_data(gm_image)
        if not shown:
            plt.show()
        shown = True
        print("The following actions are available:")
        enabled_nums = [i for i, (a, m) in enumerate(state.acts_enabled) if m]
        for i, act in enumerate(enabled):
            s = str(act)
            if act.unique_ident in greens:
                s = crayons.green(s)
            print('  [%d] %s' % (i, s))
        action_num = None
        while action_num is None:
            action_num_s = input(
                "Select your action (0-%d, 'g'/'G' to invoke generalised "
                "planner): " % (len(enabled) - 1))
            if action_num_s.strip().lower() == 'g':
                action_num = 'g'
                break
            try:
                action_num = int(action_num_s.strip())
                if not (0 <= action_num < len(enabled)):
                    raise ValueError("out of range")
            except ValueError as ex:
                print("Invalid selection (%s)" % ex)
                continue
        if action_num == 'g':
            idents = [a.unique_ident for a, en in state.acts_enabled if en]
            # XXX remove these prints
            print("Enabled actions:", ", ".join(idents))
            q_values = ds_teacher.q_values(state, idents)
            print(
                "Q-values:",
                ", ".join("%s=%g" % (ident, val)
                          for ident, val in zip(idents, q_values)))
            print('Optimal policy envelope from current state has %d states' %
                  len(ds_teacher.extract_policy_envelope(state)))
            to_select_from_enabled = np.argmin(q_values)
            to_choose = enabled_nums[to_select_from_enabled]
        else:
            to_choose = enabled_nums[action_num]
        state, _ = sample_next_state(state, to_choose, planner_exts)


def main_plan(args, planner_exts):
    import matplotlib.pyplot as plt

    # read actions & execute them one-by-one, writing results to disk
    actions = parse_gm_plan_file(args.plan)
    if args.max_cycle_repeat > 0:
        actions, num_removed = remove_cycles(
            actions, max_cycle_len=3, max_cycle_repeats=args.max_cycle_repeat)
        if num_removed > 0:
            print("Trimmed out %d actions from cycle at end of plan"
                  % num_removed)
    state = get_init_cstate(planner_exts)
    gm_images = [render_gm_state(state)]
    for action in tqdm.tqdm(actions):
        name_to_id = {
            act.unique_ident: idx
            for idx, (act, _) in enumerate(state.acts_enabled)
        }
        to_choose = name_to_id[action]
        state, _ = sample_next_state(state, to_choose, planner_exts)
        gm_images.append(render_gm_state(state))

    # figure out output dir
    basename = osp.basename(args.plan)
    time_str = datetime.datetime.now().isoformat()
    out_dir = osp.join(args.out_dir, 'render-%s-%s' % (basename, time_str))
    os.makedirs(out_dir, exist_ok=True)
    print("Writing frames to '%s':" % out_dir)
    for step, action in enumerate(['init'] + actions):
        out_path = osp.join(out_dir, '%03i-%s.png' % (step, action))
        print("    -> Writing '%s'" % out_path)
        plt.imsave(out_path, gm_images[step], vmin=0.0, vmax=1.0)


if __name__ == '__main__':
    demo()
