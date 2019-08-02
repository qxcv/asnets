"""Interfaces for teacher planners that are used to train the ASNet. Currently
supports teachers based on SSiPP (for probabilistic or deterministic problems)
or FD (for deterministic problems)."""

from abc import ABCMeta, abstractmethod
from functools import wraps

from asnets.domain_specific import get_domain_specific_planner
from asnets.fd_interface import FDQValueCache, FDTimeout
from asnets.prof_utils import can_profile
from asnets.state_reprs import CanonicalState, successors, sample_next_state

# default maximum length for policy rollouts
DEFAULT_MAX_LEN = 100


class TeacherTimeout(Exception):
    """Generic exception to signal that teacher has timed out while trying to
    execute specified operation."""
    pass


class Teacher(metaclass=ABCMeta):
    """Common base class for teacher planners."""

    @abstractmethod
    def q_values(self, cstate, act_strs):
        """Obtain Q-values for every action (dead_end_value if disabled).

        Args:
            cstate (CanonicalState): the state `s` to compute Q-values `Q(s,a)`
                with respect to.
            act_strs ([str]): list of strings representing names of actions
                `a1, …, aN` to compute Q-values `Q(s,a1), …, Q(s,aN)` with
                respect to. Should use SSiPP/.unique_ident format (which omits
                parens, like 'move l1 l2').

        Returns:
            [float]: a list of Q-values of same length as `act_strs`, one
                corresponding to each action in the given state."""
        pass

    @abstractmethod
    def single_action_label(self, cstate):
        """Get a single action label for the current state, ideally whatever
        the planner would have outputted if just called on `cstate` in
        isolation.

        Args:
            cstate (CanonicalState): state that needs to be labelled with
                expert action.

        Returns:
            str: string action in format accepted by MDPSim, e.g. "(move shakey
                l1 l2)" (includes parens, but not commas between arguments)."""
        pass

    @abstractmethod
    def extract_policy_envelope(self, cstate):
        """Produce an iterable of states visited by the teacher policy when
        initialised in the given state.

        Args:
            cstate (CanonicalState): state for teacher to start in.

        Returns:
            [CanonicalState]: list (or other iterable) of visited canonical
                states."""
        pass

    @abstractmethod
    def expert_policy_rollout(self, cstate, *, len_bound=DEFAULT_MAX_LEN):
        f"""Perform a single rollout under the expert policy & return the
        visited states.

        On probabilistic problems, this differs from extract_policy_envelope in
        two important respects: (1) it can obviously return a smaller set of
        states that may vary between calls, and (2) on probabilistic problems,
        it is actually capable of setting history features correctly on
        returned CStates.

        Args:
            cstate (CanonicalState): state in which to start rollout.
            len_bound (int): a hint to the planner that trajectories should not
                be longer than this. Useful for probabilistic planners that
                might generate arbitrarily long rollouts on some problems (e.g
                PBW, xenotravel), even under an optimal policy. Default is
                {DEFAULT_MAX_LEN}.

        Returns:
            [CanonicalState]: list or iterable of states visited during the
                rollout."""
        pass

    @property
    @abstractmethod
    def dead_end_value(self):
        """Cutoff value (float) for maximum length of a path to reach goal; no
        valid path returned by the planner has higher cost than this."""
        pass


class SSiPPTeacher(Teacher):
    """Wraps raw SSiPP planners (VI, LRTDP, [L]SSiPP, etc.)"""

    def __init__(self,
                 planner_exts,
                 planner_name,
                 heuristic_name,
                 *,
                 timeout_s=1800):
        self._plan_exts = planner_exts
        self.problem = planner_exts.ssipp_problem
        self._ssipp = planner_exts.ssipp
        self.ssp = self._ssipp.SSPfromPPDDL(self.problem)
        heuristic = self._ssipp.createHeuristic(self.ssp, heuristic_name)
        self.planner = self._ssipp.createPlanner(self.ssp, planner_name,
                                                 heuristic)
        assert timeout_s > 0, "can't have 0 timeout (got %s)" % timeout_s
        self.timeout_us = max(1, int(timeout_s * 1e6))
        # Q-value cache
        self._qv_cache = {}
        # blacklist of states that cause timeouts
        self._state_blacklist = set()

    def _set_deadline(self):
        self._ssipp.removeDeadline()
        deadline = self._ssipp.CpuTimeDeadline(self.timeout_us)
        self._ssipp.setDeadline(deadline)

    def _clear_deadline(self):
        self._ssipp.removeDeadline()

    def _blacklist_decorator(inner_func):
        """Wraps a method of this class in a SSiPP timeout, so that planning
        can't take too long. Maintains a blacklist of states that caused
        timeouts before, and prevents relevant methods from being called again
        on those states by simply raising another timeout."""

        @wraps(inner_func)
        def wrapper_func(self, cstate, *args, **kwargs):
            assert isinstance(cstate, CanonicalState), \
                "func %s was expecting a cstate but got '%r'" \
                % (inner_func, cstate)
            if cstate._ident_tup() in self._state_blacklist:
                raise TeacherTimeout(
                    'SSiPP timed out on this before; skipping')
            try:
                self._set_deadline()
                return inner_func(self, cstate, *args, **kwargs)
            except self._ssipp.DeadlineReachedException as ex:
                self._state_blacklist.add(cstate._ident_tup())
                raise TeacherTimeout('SSiPP timeout: %s' % ex)
            finally:
                self._clear_deadline()

        return wrapper_func

    @_blacklist_decorator
    def q_values(self, cstate, act_strs):
        state = cstate.to_ssipp(self._plan_exts)

        rv = []
        for act_str in act_strs:
            # cstates are great because I can hash them :)
            key = (cstate, act_str)
            if key not in self._qv_cache:
                action = self.problem.find_action(act_str)
                if action is None:
                    raise ValueError("Couldn't find action %r" % act_str)
                self._qv_cache[key] = self.planner.q_value(state, action)
            rv.append(self._qv_cache[key])

        return rv

    @_blacklist_decorator
    def single_action_label(self, cstate):
        raise NotImplementedError("this is going to be minor pain for SSiPP")

    @_blacklist_decorator
    def extract_policy_envelope(self, cstate):
        ssipp_state = cstate.to_ssipp(self._plan_exts)
        pol_dict, overflow = self._ssipp.extract_policy(
            self.planner, self.ssp, ssipp_state)
        # FIXME(history): ensure that this maintains history features. If
        # that's not possible, then write another method that just rolls out
        # single trajectories under the optimal policy, instead of computing
        # a full optimal policy envelope.
        if overflow:
            print("WARNING: extract_policy exceeded max returned state "
                  "count!")
        rv = []
        for new_ssipp_state, _ in pol_dict.items():
            new_cstate = cstate.from_ssipp(new_ssipp_state, self._plan_exts)
            rv.append(new_cstate)
        return rv

    @_blacklist_decorator
    def expert_policy_rollout(self, cstate, *, len_bound=DEFAULT_MAX_LEN):
        # TODO: see if I can get callers to set a reasonable max_len
        cstates = []
        while not cstate.is_terminal and len(cstates) < len_bound:
            ssipp_state = cstate.to_ssipp(self._plan_exts)
            ssipp_act = self.planner.decideAction(ssipp_state)
            if ssipp_act is None:
                break
            act_label = ssipp_act.name().strip('()')
            act_id = self._plan_exts.problem_meta.act_unique_id_to_index(
                act_label)
            # by appending before computing successor, we ensure that we do not
            # get the last (probably terminal) cstate in the sequence
            cstates.append(cstate)
            cstate, _ = sample_next_state(cstate, act_id, self._plan_exts)
        return cstates

    @property
    def dead_end_value(self):
        return self._plan_exts.ssipp_dead_end_value


class FDTeacher(Teacher):
    """Wraps FD as a teacher planner."""

    def __init__(self, planner_exts, *, timeout_s=1800):
        self._plan_exts = planner_exts
        self._qv_cache = FDQValueCache(planner_exts, timeout_s=timeout_s)

    def q_values(self, cstate, act_strs):
        try:
            q_values_fd = self._qv_cache.compute_q_values(cstate, act_strs)
        except FDTimeout as ex:
            raise TeacherTimeout("FDTimeout: %s" % ex)
        rv = []
        # replace the None values (for disabled actions) with dead-end values
        for qv in q_values_fd:
            if qv is None:
                rv.append(self.dead_end_value)
            else:
                rv.append(qv)
        return rv

    def single_action_label(self, cstate):
        try:
            value, action_sans_parens \
                = self._qv_cache.compute_state_value_action(cstate)
        except FDTimeout as ex:
            raise TeacherTimeout("FDTimeout: %s" % ex)
        if action_sans_parens is None:
            return None
        return '(%s)' % (action_sans_parens, )

    def expert_policy_rollout(self, cstate, *, len_bound=DEFAULT_MAX_LEN):
        return self.extract_policy_envelope(cstate)

    def extract_policy_envelope(self, cstate):
        try:
            return self._qv_cache.compute_policy_envelope(cstate)
        except FDTimeout as ex:
            raise TeacherTimeout("FDTimeout: %s" % ex)

    @property
    def dead_end_value(self):
        return self._plan_exts.ssipp_dead_end_value


class DomainSpecificTeacher(Teacher):
    """Encapsulates a domain-specific teacher planner. Works only for
    deterministic problems (general SSPs would be a bit more painful)."""

    def __init__(self, planner_exts):
        self._plan_exts = planner_exts
        domain_name = planner_exts.domain_meta.name
        self._ds_planner = get_domain_specific_planner(domain_name)
        # Maps cstates to (value, action) pairs; for dead ends (under our GP)
        # the value is set to self.dead_end_value and the action is set to
        # None. For goal states, the value is set to 0 and the action is also
        # set to None.
        self._value_action_cache = {}
        # FIXME: make the above an LRU cache so that we don't run out of memory
        # on problems with lots of states.

    def _act_id(self, act_str):
        """Turn a unique action identifier string (i.e SSiPP-formatted action
        string) into an index into the action list for this problem."""

        # FIXME: instead of doing this shit I should just fix the things in
        # state_reprs.py so that they can take action IDs (numeric ones) OR
        # action names OR BoundActions! Have a family of resolve_action_to_*
        # methods that let you convert from any one representation to any
        # other. Right now it seems like I have the same problem with actions
        # that I used to have with states (too many representations, have to be
        # careful which one you use).
        return self._plan_exts.problem_meta.act_unique_id_to_index(act_str)

    @can_profile
    def state_value_action(self, cstate):
        orig_cstate = cstate
        if cstate.is_goal:
            return (0, None)

        cached_value = self._value_action_cache.get(cstate)
        if cached_value is not None:
            value, action = cached_value
            return value, action

        plan_fragment = self._ds_planner(cstate)

        if plan_fragment is None:
            # plan failure
            return (self.dead_end_value, None)

        if len(plan_fragment) == 0:
            # this should only happen in goal states, so if it happens here
            # then that's bad
            raise ValueError(
                "DS planner %s returned empty plan for cstate, but it's "
                "not a goal (state: %s)" % (self._ds_planner, cstate))

        visited_cstates = []
        taken_actions = []
        for action_ident_to_apply in plan_fragment:
            assert not cstate.is_goal

            # check cache and break early if we get to a state with cached
            # value
            cached_va = self._value_action_cache.get(cstate)
            if cached_va is not None:
                leaf_value, _ = cached_va
                break

            # add state & action to trajectory
            visited_cstates.append(cstate)
            taken_actions.append(action_ident_to_apply)

            # get next state
            act_id = self._act_id(action_ident_to_apply)
            succs = successors(cstate, act_id, self._plan_exts)
            if len(succs) != 1:
                raise NotImplementedError(
                    "cannot deal with >1 successors (grr)")
            (_, cstate), = succs
        else:
            # if we didn't break early then we need to compute value of final
            # state
            leaf_value, _ = self.state_value_action(cstate)

        # go back over every state we visited & cache its value & optimal
        # action
        assert len(visited_cstates) == len(taken_actions)
        reverse_sa = zip(visited_cstates[::-1], taken_actions[::-1])
        for extra_cost, (cstate, action) in enumerate(reverse_sa, start=1):
            total_cost = leaf_value + extra_cost
            if total_cost >= self.dead_end_value:
                self._value_action_cache[cstate] = (self.dead_end_value, None)
            else:
                self._value_action_cache[cstate] = (total_cost, action)

        # every visited state, every action taken
        return self._value_action_cache[orig_cstate]

    def single_action_label(self, cstate):
        _, act_sans_parens = self.state_value_action(cstate)
        if act_sans_parens is None:
            return None
        return '(%s)' % act_sans_parens

    @can_profile
    def q_values(self, cstate, act_strs):
        q_values = []

        for act_str in act_strs:
            # strip() is necessary because this gets passed actions with ")"
            # and "(" in their names
            act_id = self._act_id(act_str.strip('()'))
            succs = successors(cstate, act_id, self._plan_exts)
            assert len(succs) == 1, \
                "got %d succs for %s but expected one (in state %s)" \
                % (len(succs), act_str, cstate)
            (_, next_cstate), = succs
            state_value, _ = self.state_value_action(next_cstate)

            assert state_value is not None
            # assume action cost is 1 always
            q_values.append(min(1 + state_value, self.dead_end_value))

        return q_values

    @can_profile
    def extract_policy_envelope(self, cstate):
        # Important: this only handles deterministic problems, where it
        # suffices to just make a single list of states we visit when blindly
        # executing given actions in order. Will fail if it finds something
        # that's not deterministic.

        visited = [cstate]
        init_cost, _ = self.state_value_action(cstate)

        if init_cost >= self.dead_end_value:
            # if the initial cost is >= dead_end_value then this must be dead
            # end, so we return immediately
            return visited

        while not cstate.is_goal:
            cost_to_go, action_str = self.state_value_action(cstate)
            assert action_str is not None, \
                "shouldn't have dead ends here b/c init_cost<dead_end_value"
            act_id = self._act_id(action_str)
            succs = successors(cstate, act_id, self._plan_exts)
            if len(succs) != 1:
                raise NotImplementedError("this is deterministic-only!")
            (_, cstate), = succs
            visited.append(cstate)

        return visited

    def expert_policy_rollout(self, cstate, *, len_bound=DEFAULT_MAX_LEN):
        return self.extract_policy_envelope(cstate)

    @property
    def dead_end_value(self):
        return self._plan_exts.ssipp_dead_end_value
