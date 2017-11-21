"""Extra padding around SSIPP to make it a little less likely to hurt me."""

import re
from typing import Dict, Tuple, List  # noqa


def set_up_ssipp(ssipp_module, pddl_paths, problem_name):
    """Parse some PDDL and fetch a problem."""
    for file_path in pddl_paths:
        ssipp_module.readPDDLFile(file_path)
    problem = ssipp_module.init_problem(problem_name)

    return problem


class Evaluator:
    def __init__(self, planner_exts, heuristic_name):
        self._ssipp = planner_exts.ssipp
        self.problem = planner_exts.ssipp_problem
        ssp = self._ssipp.SSPfromPPDDL(self.problem)
        heuristic = self._ssipp.createHeuristic(ssp, heuristic_name)
        self.evaluator = self._ssipp.SuccessorEvaluator(heuristic)

    def eval_state(self, state_string):
        state = self.problem.get_intermediate_state(state_string)
        return self.evaluator.state_value(state)

    def succ_probs_vals(self, state_string, action_name):
        action = self.problem.find_action(action_name)
        state = self.problem.get_intermediate_state(state_string)
        return [(e.probability, e.value)
                for e in self.evaluator.succ_iter(state, action)]

    def dump_table(self, path):
        self.evaluator.dump_table(path)


class Cutter:
    # ssipp appends -prob-j to an action name to signify that it is the j-th
    # (determinised) outcome of the original action. -prob-j-prec-i is used for
    # the j-th determinised outcome of the i-th (disjunctive) case.

    act_re = re.compile(r'^(\(.+?\))(?:-(?:prob|prec|c)-\d+)*$')

    def __init__(self, planner_exts):
        self.problem = planner_exts.ssipp_problem
        self.lm_cut = planner_exts.ssipp.LMCutHeuristic(self.problem)

    def real_action_name(self, act_name):
        match = self.act_re.match(act_name)
        if match is None:
            raise ValueError("Couldn't parse action name '%s'" % act_name)
        group, = match.groups()
        return group

    def get_action_cuts(self, state_string):
        state = self.problem.get_intermediate_state(state_string)
        cuts_value = self.lm_cut.valueAndCuts(state)
        new_cuts = []
        for cut in cuts_value.cuts:
            new_cut = frozenset(self.real_action_name(name) for name in cut)
            new_cuts.append(new_cut)
        return new_cuts


class Planner:
    """Wraps raw SSiPP planners (VI, LRTDP, [L]SSiPP, etc.)"""

    def __init__(self,
                 planner_exts,
                 planner_name: str,
                 heuristic_name: str,
                 timeout_s=1800) -> None:
        self._plan_exts = planner_exts
        self.problem = planner_exts.ssipp_problem
        self._ssipp = planner_exts.ssipp
        self.ssp = self._ssipp.SSPfromPPDDL(self.problem)
        heuristic = self._ssipp.createHeuristic(self.ssp, heuristic_name)
        self.planner = self._ssipp.createPlanner(self.ssp, planner_name,
                                                 heuristic)
        self.timeout_us = timeout_s * int(1e6)
        # Q-value cache
        self._qv_cache = {}  # type: Dict[Tuple[str, str], float]
        # cache for .deliberate()
        self._d_cache = {}  # type: Dict[str, Tuple[str, float, bool]]

    def deliberate(self, state_string: str) -> Tuple[str, float, bool]:
        """Return best action for current state, value for current state, and
        whether it is a (verified) dead end."""
        if state_string not in self._d_cache:
            self._ssipp.removeDeadline()
            deadline = self._ssipp.CpuTimeDeadline(self.timeout_us)
            self._ssipp.setDeadline(deadline)
            state = self.problem.get_intermediate_state(state_string)
            # For most planners, this will fully solve the problem from the
            # given state BEFORE figuring out best action. In some planners,
            # you can configure it to do something less intelligent (e.g.
            # extend value table with single trial of LRTDP; return greedy
            # action w.r.t. table).
            self._ssipp_act = self.planner.decideAction(state)
            state_value = self.planner.value(state)
            is_dead_end = state_value >= self._ssipp.get_dead_end_value()
            # returns something like `(name obj1 obj2 …)`
            act_rv = self._ssipp_act.name(
            ) if self._ssipp_act is not None else None
            self._d_cache[state_string] = (act_rv, state_value, is_dead_end)
        return self._d_cache[state_string]

    def q_values(self, state_string: str, act_strs: List[str]) -> List[float]:
        """Obtain Q-values for every action (dead_end_value if disabled)."""
        state = self.problem.get_intermediate_state(state_string)

        # lots of .decideAction()s in q_value thing, so could be heavy
        self._ssipp.removeDeadline()
        deadline = self._ssipp.CpuTimeDeadline(self.timeout_us)
        self._ssipp.setDeadline(deadline)

        rv = []
        for act_str in act_strs:
            key = (state_string, act_str)
            if key not in self._qv_cache:
                action = self.problem.find_action(act_str)
                if action is None:
                    raise ValueError("Couldn't find action %r" % act_str)
                self._qv_cache[key] = self.planner.q_value(state, action)
            rv.append(self._qv_cache[key])

        return rv

    def extract_policy(self, state):
        """Wrapper around ssipp.extract_policy; will return the policy as a
        state -> action dict, and also return a flag indicating whether
        extraction had to stop early because too many states were visited
        (True) or not (False)."""
        return self._ssipp.extract_policy(self.planner, self.ssp, state)

    @property
    def dead_end_value(self):
        return self._plan_exts.ssipp_dead_end_value


def format_state_for_ssipp(all_props):
    """Converts true prop list to string format that SSiPP can read.
    all_props can be obtained from an FPGObservation instance's props_true
    attributes."""
    format_props = []
    for prop_obj, truth in all_props:
        if not truth:
            continue
        old_prop = prop_obj.identifier
        assert old_prop[0] == '(', old_prop
        assert old_prop[-1] == ')', old_prop
        tokens = old_prop[1:-1].split()
        name = tokens[0]
        args = tokens[1:]
        format_props.append('%s %s' % (name, ' '.join(args)))
    return ', '.join(format_props)
