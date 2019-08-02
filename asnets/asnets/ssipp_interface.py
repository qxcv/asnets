"""Extra padding around SSIPP to make it a little less likely to hurt me."""

import importlib
import os
import re
import subprocess

ABOVE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SSIPP_MANUAL_DIR = os.path.join(ABOVE_DIR, 'ssipp-solver')


def has_ssipp_solver():
    """Check whether we have solver_ssp somewhere (this is the SSiPP planner
    binary, not the SSiPP Python library)."""
    try:
        get_ssipp_solver_path_auto()
        return True
    except (FileNotFoundError, ImportError):
        return False


def try_install_ssipp_solver():
    """Try to install SSiPP solver (i.e planner binary, not library) if it is
    not available already."""

    # TODO: make this installer less racy (& do same for try_install_fd)

    if has_ssipp_solver():
        return

    print("Installing SSiPP's solver to %s" % SSIPP_MANUAL_DIR)
    if not os.path.exists(SSIPP_MANUAL_DIR):
        subprocess.run([
            "git", "clone", "https://gitlab.com/qxcv/ssipp.git",
            SSIPP_MANUAL_DIR
        ],
                       check=True,
                       cwd=ABOVE_DIR)
        subprocess.run(["python", "build.py", "solver_ssp"],
                       check=True,
                       cwd=SSIPP_MANUAL_DIR)
    ssipp_binary_path = os.path.join(SSIPP_MANUAL_DIR, "solver_ssp")
    assert os.path.exists(ssipp_binary_path), \
        "install failed; nothing found at '%s'" % (ssipp_binary_path, )


def get_ssipp_solver_path_auto():
    """Automagically get path to SSiPP solver_ssp by assuming it's in the same
    directory as the SSiPP Python module, or that it has been downloaded into
    the current dir. Let current dir take preference."""
    # first check current dir
    current_dir_solver = os.path.join(SSIPP_MANUAL_DIR, 'solver_ssp')
    if os.path.exists(current_dir_solver):
        return current_dir_solver

    # if that failed, we check the other dir
    ssipp_spec = importlib.util.find_spec('ssipp')
    suggestion = "Maybe it's easier to compile SSiPP manually and use " \
        "--ssipp-path to specify path? Or use try_install_ssipp_solver()?"
    if ssipp_spec is None:
        raise ImportError(
            "Could not import SSiPP to do auto-magic solver_ssp path "
            "detection. " + suggestion)
    ssipp_dir = os.path.dirname(ssipp_spec.origin)
    solver_ssp_path = os.path.join(ssipp_dir, 'solver_ssp')
    if not os.path.exists(solver_ssp_path):
        raise FileNotFoundError(
            "Could not auto-magically detect SSiPP solver_ssp at '%s'. %s" %
            (solver_ssp_path, suggestion))

    return solver_ssp_path


def set_up_ssipp(ssipp_module, pddl_paths, problem_name):
    """Parse some PDDL and fetch a problem."""
    for file_path in pddl_paths:
        ssipp_module.readPDDLFile(file_path)
    problem = ssipp_module.init_problem(problem_name)

    return problem


class Evaluator:
    # FIXME: the name of this class is terrible. What does it actually do? Is
    # it just evaluating heuristics, or is it planning underneath? Resolve &
    # rename!
    def __init__(self, planner_exts, heuristic_name):
        self._ssipp = planner_exts.ssipp
        self.problem = planner_exts.ssipp_problem
        ssp = self._ssipp.SSPfromPPDDL(self.problem)
        heuristic = self._ssipp.createHeuristic(ssp, heuristic_name)
        self.evaluator = self._ssipp.SuccessorEvaluator(heuristic)

    def eval_state(self, ssipp_state):
        return self.evaluator.state_value(ssipp_state)

    def succ_probs_vals(self, ssipp_state, action_name):
        action = self.problem.find_action("(" + action_name + ")")
        assert action is not None, "could not find %r" % (action_name, )
        return [(e.probability, e.value)
                for e in self.evaluator.succ_iter(ssipp_state, action)]


class Cutter:
    # ssipp appends -prob-j to an action name to signify that it is the j-th
    # (determinised) outcome of the original action. -prob-j-prec-i is used for
    # the j-th determinised outcome of the i-th (disjunctive) case.

    act_re = re.compile(r'^(\(.+?\))(?:-(?:prob|prec|c)-\d+)*$')

    def __init__(self, planner_exts):
        self.problem = planner_exts.ssipp_problem
        self.lm_cut = planner_exts.ssipp.LMCutHeuristic(self.problem)
        # we cache cuts forever
        self.cut_cache = {}

    def real_action_name(self, act_name):
        match = self.act_re.match(act_name)
        if match is None:
            raise ValueError("Couldn't parse action name '%s'" % act_name)
        group, = match.groups()
        return group

    def get_action_cuts(self, ssipp_state):
        if ssipp_state not in self.cut_cache:
            cuts_value = self.lm_cut.valueAndCuts(ssipp_state)
            new_cuts = []
            for cut in cuts_value.cuts:
                new_cut = frozenset(
                    self.real_action_name(name) for name in cut)
                new_cuts.append(new_cut)
            self.cut_cache[ssipp_state] = new_cuts
        return self.cut_cache[ssipp_state]


def format_state_for_ssipp(all_props):
    """Converts true prop list to string format that SSiPP can read.
    all_props can be obtained from an MDPSimObservation instance's props_true
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
