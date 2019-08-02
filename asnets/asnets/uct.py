"""Code for evaluating ASNets with UCT, using ASNet policy as rollout policy.
Implemented using SSiPP as state simulator. See "Trial-based Heuristic Tree
Search for Finite Horizon MDPs" (Keller & Helmert, 2012) for algorithm
description and notation."""

# TODO: fixes to make this code more elegant.
#
# - Make "cost so far" part of the state, and push "over maximum cost" check
#   into "is this node terminal?" part of the state. This sort of behaviour
#   will have to be assured separately for each kind of environment (planning
#   environments, my gridworld test environments, etc.). That should make the
#   problem properly Markovian, and avoid issues down-the-track with my
#   reparenting code.

from abc import ABCMeta, abstractmethod
from math import sqrt, log
import random
from time import time
from warnings import warn

import numpy as np

from asnets.py_utils import weak_ref_to


def obj_array(arr):
    """Creates a Numpy object array from given iterable."""
    return np.array(tuple(arr) + (None, ))[:-1]


class UCTEnvironment(object, metaclass=ABCMeta):
    """Abstract base class for all environments used by UCT."""

    @property
    @abstractmethod
    def initial_state(self):
        """Returns abstract representation of start state."""
        pass

    @abstractmethod
    def successors(self, state, action):
        """Yields tuple of `[(prob, succ_state, cost)]` for given `state` and
        `action`, where `prob` gives probability of moving into `succ_state`
        after applying `action` in `state`."""
        pass

    @abstractmethod
    def available_actions(self, state):
        """Yields list of available actions (in whatever abstract
        representation is appropriate) for current state."""
        pass

    @abstractmethod
    def is_terminal_state(self, state):
        """Indicates whether given state is terminal (goal or failure)."""
        pass

    @abstractmethod
    def is_goal_state(self, state):
        """Indicates whether given state is a goal state (implies terminal)."""
        pass

    def print_state(self, state):
        """Write a string representation of state to the console."""
        print("State: %s" % (state, ))


class UCT(object):
    def __init__(self,
                 env,
                 policy,
                 c=1,
                 rand_seed=None,
                 dead_end_penalty=500.0):
        """
        Construct an empty tree for UCT.

        Args:
            env (UCTEnvironment): an adaptor for some environment.
            policy ((env, state) -> [(prob, action)]): given a state (in
                whatever internal representation is appropriate), this function
                yields a list indicating the transition probabilities and
                identities of each succesor state.
            c (float): c-value for use in UCB1.
            rand_seed (int, optional): seed to use for internal random
                generator. If not provided, this class uses system generator.
        """
        assert c >= 0, "can't have negative exploration param"
        self.env = env
        self.c = c
        if rand_seed is None:
            # use whatever Numpy gives us (meh)
            self.rand_gen = np.random.RandomState()
        else:
            self.rand_gen = np.random.RandomState(rand_seed)
        self.policy = policy
        # need to create this last!
        self.root = DecisionNode(
            state=env.initial_state, tree=self, parent=None)
        self.dead_end_penalty = dead_end_penalty

    @staticmethod
    def _find_decision_node(state, from_children):
        # find a decision node
        for child in from_children:
            assert isinstance(child, DecisionNode)
            if child.state == state:
                return child
        raise ValueError("could not find state %s among children %s" %
                         (state, from_children))

    @staticmethod
    def _find_chance_node(action, from_children):
        for child in from_children:
            assert isinstance(child, ChanceNode)
            if child.action == action:
                return child
        raise ValueError("could not find action %s among children %s" %
                         (action, from_children))

    def reparent_tree_downwards(self, chosen_action, result_state):
        """Change root of tree after an environment transition. This prunes the
        tree so as to emulate choosing action `chosen_action` (in abstract
        representation), then landing in one of its children `result_state` (in
        abstract representation)."""
        assert isinstance(self.root, DecisionNode)
        for child in self.root.children:
            if child.action == chosen_action:
                chance_node = child
                break
        else:
            raise RuntimeError("Could not find action %r" % (chosen_action, ))
        assert isinstance(chance_node, ChanceNode)
        for grandchild in chance_node.children:
            if grandchild.state == result_state:
                new_root = grandchild
                break
        else:
            raise RuntimeError("child %r has no grandchild for state %r" %
                               (chance_node, result_state))
        self.root = new_root
        # ensure we're marked as root
        self.root.parent = None

    def _do_rollout(self):
        current_node = self.root
        cost = 0
        cumulative_costs = [cost]

        # do tree portion
        while current_node.is_expanded and not current_node.is_terminal \
                and cost < self.dead_end_penalty:
            current_node, this_cost = current_node.select_child()
            cost += this_cost
            cumulative_costs.append(cost)

        while not isinstance(current_node, (DecisionNode, LeafDecisionNode)):
            # jump forward to a decision node
            current_node.expand()
            current_node, this_cost = current_node.select_child()
            cost += this_cost
            cumulative_costs.append(this_cost)

        if not current_node.is_expanded:
            # make sure we expand here so that we go one deeper next time
            current_node.expand()

        rollout_state = current_node.state

        while not self.env.is_terminal_state(rollout_state):
            policy_succs = self.policy(self.env, rollout_state)
            policy_dist, policy_acts = zip(*policy_succs)
            # choose an action using rollout policy
            chosen_act = self.rand_gen.choice(
                obj_array(policy_acts), p=policy_dist)
            # choose a successor randomly
            # TODO: support rollout policies!
            env_succs = self.env.successors(rollout_state, chosen_act)
            env_dist = [p for p, _, _ in env_succs]
            _, rollout_state, this_cost = self.rand_gen.choice(
                obj_array(env_succs), p=env_dist)
            cost += this_cost

        if self.env.is_terminal_state(rollout_state) \
           and not self.env.is_goal_state(rollout_state):
            # dead end
            cost = self.dead_end_penalty

        # we can't exceed the dead end penalty
        cost = min(cost, self.dead_end_penalty)

        # now do MC backup in reverse order
        while current_node is not None:
            future_ret = cost - cumulative_costs.pop()
            current_node.backup_update(future_ret)
            current_node = current_node.parent

        assert len(cumulative_costs) == 0, \
            "some remaining costs: %r" % cumulative_costs

    def choose_action(self, *, max_rollouts=None, max_time=None):
        if max_rollouts is None and max_time is None:
            max_rollouts = 100
        num_rollouts = 0
        start = time()
        while True:
            self._do_rollout()
            num_rollouts += 1
            elapsed = time() - start
            if max_rollouts is not None and num_rollouts >= max_rollouts:
                break
            if max_time is not None and elapsed >= max_time:
                break
        return self.root.choose_action_final()

    def _print_subtree_lines(self, node):
        this_rep = node.str_node()
        if node.is_terminal or not node.is_expanded:
            if node.is_goal:
                this_rep += ' (goal)'
            elif node.is_terminal:
                this_rep += ' (dead end)'
            else:
                this_rep += ' (leaf)'
            lines = [this_rep]
        else:
            lines = [this_rep]
            indent = ' | '
            for child in node.children:
                child_lines = self._print_subtree_lines(child)
                lines.extend(indent + line for line in child_lines)
        return lines

    def str_tree(self, node=None):
        """String representation of tree."""
        if node is None:
            node = self.root
        lines = self._print_subtree_lines(node)
        return "\n".join(lines)


class UCTNode(object, metaclass=ABCMeta):
    def __init__(self, tree, parent=None):
        # weakrefs to avoid GC cycles (where we point to tree & its members,
        # while tree also points to us)
        self.tree = weak_ref_to(tree)
        self.env = weak_ref_to(tree.env)
        self.policy = weak_ref_to(tree.policy)
        self.parent = weak_ref_to(parent)
        self._children = None

    @property
    def children(self):
        """Return children of node."""
        assert self.is_expanded, "tried to use children of unexpanded node"
        return self._children

    @property
    def is_expanded(self):
        """Has this node had its children assigned?"""
        return self._children is not None

    @property
    @abstractmethod
    def is_terminal(self):
        """Is this node going to have any children?"""
        pass

    @property
    @abstractmethod
    def is_goal(self):
        """Is this node a goal node?"""
        pass

    @abstractmethod
    def select_child(self):
        """Sample a child & cost for rollout purposes."""
        pass

    @abstractmethod
    def expand(self):
        """Give this node some (unexpanded) children."""
        pass

    @abstractmethod
    def backup_update(self, ret):
        """Do a backup for a single rollout."""
        pass

    @abstractmethod
    def str_node(self):
        """Human-readable representation of node."""
        pass


class DecisionNode(UCTNode):
    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # what state are we associated  with?
        self.state = state
        self.backup_count = 0
        self.mean_cost = None
        if self.env.is_terminal_state(self.state):
            raise TypeError("Use LeafDecisionNode for terminal states!")

    def ucb1(self, child):
        """Calculate UCB1 formula value for child (infinite if child is
        unvisited and c > 0)."""
        if child.backup_count == 0 and self.tree.c > 0:
            # in this case we need to pick an unvisited node!
            return float('inf')
        if self.backup_count == 0 and child.backup_count > 0:
            raise RuntimeError(
                "Parent count %d, but child count %d? There's a bug somewhere."
                % (self.backup_count, child.backup_count))
        explore = sqrt(2 * log(self.backup_count) / child.backup_count)
        explore_term = self.tree.c * explore
        cost_term = -child.mean_cost
        ucb1 = cost_term + explore_term
        return ucb1

    @property
    def is_terminal(self):
        return False

    @property
    def is_goal(self):
        return self.env.is_goal_state(self.state)

    def select_child(self):
        # associate cost with chance nodes, not decision nodes
        cost = 0
        # we shuffle children so that we pick randomly when we have an
        # unvisited child
        shuf_children = list(self.children)
        self.tree.rand_gen.shuffle(shuf_children)
        return max(shuf_children, key=self.ucb1), cost

    def choose_action_final(self):
        # don't bother using UCB1, just choose action with best cost
        if not any(c.backup_count > 0 for c in self.children):
            warn('Decision node %r has no visited children at decision-time!' %
                 (self, ))
        best_child = min(self.children, key=lambda c: c.mean_cost)
        return best_child.action

    def backup_update(self, ret):
        # back-up just means choosing best child
        # FIXME should be possible to do this faster by just looking at the
        # child that has changed
        self.backup_count += 1
        args = [c.mean_cost for c in self.children if c.backup_count > 0]
        if not args:
            # this is not very elegant, but I'm not sure how to handle tip
            # nodes in an elegant way
            self.mean_cost = ret
        else:
            self.mean_cost = min(args)

    def expand(self):
        child_acts = self.env.available_actions(self.state)
        children = []
        for action in child_acts:
            new_node = ChanceNode(action=action, tree=self.tree, parent=self)
            children.append(new_node)
        self._children = children

    def str_node(self):
        return "Decision in state %s (N=%d)" % (self.state, self.backup_count)


class LeafDecisionNode(UCTNode):
    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # what state are we associated  with?
        self.state = state
        self.backup_count = 0
        self.mean_cost = None
        if not self.env.is_terminal_state(self.state):
            raise TypeError("This class is only for terminal states")

    @property
    def is_terminal(self):
        return True

    @property
    def is_goal(self):
        return self.env.is_goal_state(self.state)

    def select_child(self):
        raise TypeError("Node is terminal, can't sample child")

    def choose_action_final(self):
        raise TypeError("Node is terminal, can't choose an action child")

    def backup_update(self, ret):
        # just use actual return as value estimate
        self.backup_count += 1
        self.mean_cost = ret

    def expand(self):
        self._children = []

    def str_node(self):
        return "Decision in terminal state %s (N=%d, V=%s)" \
            % (self.state, self.backup_count, self.mean_cost)


class ChanceNode(UCTNode):
    def __init__(self, action, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # what action are we associated with?
        self.action = action
        # backup_count corresponds to N(v) in source paper
        self.backup_count = 0
        # source paper uses Q(v) to refer to *sum* of rewards, but we store
        # only the mean. That means self.mean_cost * self.backup_count =
        # [Q(v)/N(v)] * N(v) = Q(v). Also note that I'm using Q(v) to mean
        # 'cost' rather than 'reward'.
        self.mean_cost = None
        # make sure there's a state node above us
        assert isinstance(self.parent, DecisionNode), \
            "need a decision node above! got %r instead" % self.parent

    def select_child(self):
        rand_gen = self.tree.rand_gen
        chosen_idx = rand_gen.choice(
            self._children_inds, p=self._children_dist)
        chosen = self._children[chosen_idx]
        # THTS paper associates costs with chance nodes rather than decision
        # nodes, and I think that makes a lot of sense from an implementation
        # perspective.
        cost = self._children_costs[chosen_idx]
        return chosen, cost

    @property
    def is_terminal(self):
        # chance nodes cannot be terminal
        return False

    @property
    def is_goal(self):
        # chance nodes cannot be goals
        return False

    def backup_update(self, ret):
        """Update just this node as part of a backup"""
        self.backup_count += 1
        # do a full Bellman backup (per THTS paper)
        tot_prob = 0.0
        exp_cost = 0.0
        for child, transition_cost, p in zip(
                self.children, self._children_costs, self._children_dist):
            if child.backup_count == 0:
                continue
            tot_prob += p
            exp_cost += p * (transition_cost + child.mean_cost)
        assert tot_prob > 0, "did a backup with no visited children (?)"
        assert tot_prob <= 1 + 1e-5, "uuuuuh 1 < %f" % tot_prob
        self.mean_cost = exp_cost / tot_prob

    def expand(self):
        state = self.parent.state
        succ_states = self.env.successors(state, self.action)
        children = []
        children_dist = []
        children_costs = []
        # hmm, we're not using probabilities. surely there's more efficient way
        # to do this?
        for p, state, cost in succ_states:
            if self.env.is_terminal_state(state):
                new_node = LeafDecisionNode(
                    state=state, tree=self.tree, parent=self)
            else:
                new_node = DecisionNode(
                    state=state, tree=self.tree, parent=self)
            children.append(new_node)
            children_dist.append(p)
            children_costs.append(cost)
        assert len(children) == len(children_dist)
        assert len(children) == len(children_costs)
        self._children = children
        # want these to be NP arrays so that we can use np.random.choice
        self._children_dist = np.asarray(children_dist)
        self._children_inds = np.arange(len(children))
        self._children_costs = children_costs

    def str_node(self):
        return "Chance outcomes of %s (Q=%s, N=%d)" \
            % (self.action,
               None if self.mean_cost is None else '%.2f' % self.mean_cost,
               self.backup_count)


# ########################################################################## #
# #####################                                ##################### #
# ##################### Rest of this code is test code ##################### #
# #####################                                ##################### #
# ########################################################################## #


class WindyGridworld(UCTEnvironment):
    """Simulator for a simple gridworld in which wind blows from top to bottom,
    and there is a cliff at the bottom. Start and goal are just above the
    cliff, with straight (but dangerous) line connecting them.

    |-|-|…|-|-|
    |…|…|…|…|…|  Empty grid squares with wind blowing downward
    |-|-|…|-|-|
    |S|-|…|-|G|  Start/goal (empty, but windy, grid squares between)
    |-|C|…|C|-|  Cliff danger zone!
    """
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'

    DELTAS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

    def __init__(self, width, height, wind_p=0.3, move_cost=1):
        assert width >= 2, "need enough space for start/goal"
        assert height >= 2, "can't "
        self.width = width
        self.height = height
        self.move_cost = move_cost
        self.wind_p = wind_p
        # (x, y) with zero-based indices from bottom left
        self.start_pos = (0, 1)
        self.goal_pos = (width - 1, 1)

    @property
    def initial_state(self):
        return self.start_pos

    def is_terminal_state(self, state):
        return self.is_goal_state(state) or state[1] == 0

    def is_goal_state(self, state):
        return state == self.goal_pos

    def successors(self, state, action):
        # produces list of (prob, succ state)
        if action not in self.available_actions(state):
            raise ValueError('action %s is invalid in state %s' %
                             (action, state))
        x, y = state
        dx, dy = self.DELTAS[action]
        # might take desired action
        no_gust_state = x + dx, y + dy
        # but there's also a chance we'll be blown off normal course
        gust_state = x + dx, y + dy - 1
        if action == self.DOWN:
            # 100% reliable
            return [(1, no_gust_state, self.move_cost)]
        else:
            return [(1 - self.wind_p, no_gust_state, self.move_cost),
                    (self.wind_p, gust_state, self.move_cost)]

    def available_actions(self, state):
        # if we're in a terminal state then we can't do anything
        if self.is_terminal_state(state):
            return []
        # otherwise, we can choose from one of several actions
        from_left, from_bot = state
        acts = []
        if from_left > 0:
            acts.append(self.LEFT)
        if from_left < self.width - 1:
            acts.append(self.RIGHT)
        if from_bot > 0:
            acts.append(self.DOWN)
        if from_bot < self.height - 1:
            acts.append(self.UP)
        return acts

    def format_state(self, state):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        # this is the cliff (y-flipped)
        for col in range(self.width):
            grid[0][col] = 'C'
        # this is where we start (y-flipped)
        xs, ys = self.start_pos
        grid[ys][xs] = 'S'
        # this is where we end (y-flipped)
        xg, yg = self.goal_pos
        grid[yg][xg] = 'G'
        # this is where we are (y-flipped)
        x, y = state
        grid[y][x] = 'P'
        # now unflip
        grid = grid[::-1]
        # borders
        grid = [[' |'] + g + ['|'] for g in grid]
        pre_pad = [' +'] + ['-'] * self.width + ['+']
        grid.insert(0, pre_pad)
        grid.append(pre_pad)
        grid_part = '\n'.join(''.join(l) for l in grid)
        # add stats
        stats_line = 'wind_p = %.2g, move_cost = %.2g\n' % (self.wind_p,
                                                            self.move_cost)
        return stats_line + grid_part

    def print_state(self, state):
        print(self.format_state(state))


def print_random_trajectory():
    gw = WindyGridworld(8, 5)
    state = gw.initial_state
    gw.print_state(state)
    print()
    while not gw.is_terminal_state(state):
        avail_acts = gw.available_actions(state)
        if gw.DOWN in avail_acts and state[0] < gw.width - 1 \
           and len(avail_acts) > 1:
            # don't go down if we can help it
            avail_acts.remove(gw.DOWN)
        action = random.choice(avail_acts)
        print('Go %s' % action)
        succs = gw.successors(state, action)
        probs, succ_states, costs = zip(*succs)
        # we need this hackery to make a 1D array of (int, int) tuples, instead
        # of a 2D array of ints
        succ_array = obj_array(succ_states)
        state = np.random.choice(succ_array, p=probs)
        gw.print_state(state)
        print()
    if gw.is_goal_state(state):
        print('GOOOOOAL!')
    else:
        print('Failure :(')


def make_ssp_env(*args, **kwargs):
    # build this class inside a function so that we can hide imports (this way
    # we can run uct.py in isolation, without importing stuff from other
    # modules)
    from state_reprs import CanonicalState, get_init_cstate

    class SSPEnvironment(UCTEnvironment):
        """Allows UCT to operate on PPDDL-style SSP environments."""

        # TODO: cache these (somewhat expensive) methods!

        def __init__(self, planner_exts, heuristic_name='h-add'):
            # WARNING: creating planner_exts like this is not safe to do in
            # master process! A PlannerExtensions object should generally not
            # leave the subprocess for the associated problem.
            self.p = planner_exts
            self._ssipp_problem = planner_exts.ssipp_problem
            ssp = self.p.ssipp.SSPfromPPDDL(self._ssipp_problem)
            heuristic = self.p.ssipp.createHeuristic(ssp, heuristic_name)
            self.evaluator = self.p.ssipp.SuccessorEvaluator(heuristic)

        @property
        def initial_state(self):
            return get_init_cstate(self.p)

        def successors(self, cstate, action_name):
            ssipp_state = cstate.to_ssipp(self.p)
            action = self._ssipp_problem.find_action("(" + action_name + ")")
            cost = action.cost(ssipp_state)
            # use e.value to get heuristic values
            rv = [(e.probability, CanonicalState.from_ssipp(e.state,
                                                            self.p), cost)
                  for e in self.evaluator.succ_iter(ssipp_state, action)]
            return rv

        def available_actions(self, cstate):
            return [a.unique_ident for a, t in cstate.acts_enabled if t]

        def is_terminal_state(self, cstate):
            return len(self.available_actions(cstate)) == 0

        def is_goal_state(self, cstate):
            return cstate.is_goal

    return SSPEnvironment(*args, **kwargs)


def do_uct_demo(env='bus'):
    # time to deliberate for between choosing actions
    wait_time = 5
    if env == 'bus':
        # Demo using SSiPP + UCT + ASNets together
        from supervised import PlannerExtensions
        from models import PropNetwork, PropNetworkWeights
        import tensorflow as tf

        # set up env first
        exts = PlannerExtensions(
            ['../problems/little-thiebaux/interesting/bus-fare.pddl'],
            'bus-fare-problem')
        # more complex (TTW):
        # exts = PlannerExtensions(
        #     [
        #         '../problems/little-thiebaux/interesting/triangle-tire.pddl',
        #         '../problems/little-thiebaux/interesting/triangle-tire-small.pddl'
        #     ],
        #     'triangle-tire-1')
        env = make_ssp_env(exts)

        # now set up network (a bit complicated; check run_asnets.py to see how
        # this is set up normally)
        dg_extra_dim = sum(dg.extra_dim for dg in exts.data_gens)
        sess = tf.Session()
        weights = PropNetworkWeights(
            exts.domain_meta,
            hidden_sizes=[(4, 4)] * 2,
            extra_dim=dg_extra_dim,
            skip=True)
        network = PropNetwork(weights, exts.problem_meta, dropout=0.0)
        sess.run(tf.global_variables_initializer())

        def policy(ssp_env, cstate):
            """Use the (untrained) action schema network to choose an
            action."""

            # first a dead end sanity check
            allowed = env.available_actions(cstate)
            if not allowed:
                raise ValueError(
                    'Asked policy for action in terminal state (%r)' % cstate)

            # input_batch is an actual numpy ndarray that we can give to the
            # network!
            input_batch = cstate.to_network_input()
            # add leading (batch) dimension for network
            input_batch = input_batch[None]
            action_dist_batch = sess.run(
                network.act_dist, feed_dict={network.input_ph: input_batch})
            # result is a single vector representing a probability distribution
            # over actions
            action_dist = action_dist_batch[0]

            # we need to match action numbers up to names, since we should be
            # returning list of [(probability, action name)] (in this case)
            return_dist = []
            allowed_set = set(allowed)
            dist_sum = 0
            for act_num, p in enumerate(action_dist):
                action_name = cstate.acts_enabled[act_num][0].unique_ident
                if action_name in allowed_set:
                    return_dist.append((p, action_name))
                    dist_sum += p

            # renormalise because np.random.choice is really picky for whatever
            # reason
            assert abs(dist_sum - 1) < 1e-2
            return_dist = [(p / dist_sum, a) for p, a in return_dist]

            return return_dist

    elif env == 'windy':
        env = WindyGridworld(4, 4)

        def policy(env, state):
            """Garbage rollout policy that just chooses actions at random."""
            allowed = env.available_actions(state)
            if not allowed:
                raise ValueError(
                    'Asked policy for action in terminal state (%r)' % state)
            p = 1.0 / len(allowed)
            return [(p, a) for a in allowed]
    else:
        raise ValueError("Unknown environment %s" % (env, ))

    # c and dead end penalty shuld be roughly balanced
    tree = UCT(env, policy, rand_seed=None, c=100, dead_end_penalty=100)

    state = tree.root.state
    print()
    while True:
        print('Waiting for action...', end='')
        action = tree.choose_action(max_time=wait_time)
        print(tree.str_tree())
        print(' chose %r' % (action, ))
        succs = env.successors(state, action)
        probs, succ_states, costs = zip(*succs)
        succ_array = obj_array(succ_states)
        state = np.random.choice(succ_array, p=probs)
        env.print_state(state)
        if not env.is_terminal_state(state):
            # re-use the same tree to choose our next action
            tree.reparent_tree_downwards(action, state)
        else:
            break

    if env.is_goal_state(state):
        print('GOOOOOAL!')
    else:
        print('Failure :(')


def main():
    # print_random_trajectory()
    do_uct_demo()


if __name__ == '__main__':
    main()
