"""Tool for interfacing to Fast Downward (FD) planner, for deterministic
problems. Based on `run_det_baselines.py` from `../det-baselines/` (but with
some changes)."""

import argparse
import contextlib
import os
import os.path as osp
import re
import shutil
import subprocess
import time
import uuid

import numpy as np

from asnets.pddl_utils import extract_domain_problem, hlist_to_sexprs, \
    replace_init_state
from asnets.state_reprs import sample_next_state, get_init_cstate

THIS_DIR = osp.dirname(osp.abspath(__file__))
ABOVE_DIR = osp.abspath(osp.join(THIS_DIR, '..'))
FD_DIR = osp.join(ABOVE_DIR, 'fd')
FD_PATH = osp.join(FD_DIR, 'fast-downward.py')
PLAN_BN = 'plan.out'
STDOUT_BN = 'stdout.txt'
STDERR_BN = 'stderr.txt'
CMDLINE_BN = 'cmdline.txt'


def has_fd():
    return osp.exists(FD_PATH)


def try_install_fd():
    # If necessary, installs Fast Downward per
    # http://www.fast-downward.org/ObtainingAndRunningFastDownward

    # TODO: make this less racy; should be okay to call it from several
    # processes at once.
    if not has_fd():
        print("Installing FD")
        if not osp.exists(FD_DIR):
            # these commands check out a specific revision of fd, per
            # https://stackoverflow.com/a/3489576
            subprocess.run(["git", "init", "fd"], check=True, cwd=ABOVE_DIR)
            subprocess.run([
                "git", "remote", "add", "origin",
                "https://github.com/aibasel/downward.git"
            ], check=True, cwd=FD_DIR)
            subprocess.run(
                ["git", "fetch", "origin",
                 # most recent revision last time this script was updated
                 # (March 2022)
                 "f42dfc992df1ce5a65312c0eeebaf7236e8ffdf8"],
                check=True, cwd=FD_DIR)
            subprocess.run(
                ["git", "reset", "--hard", "FETCH_HEAD"],
                check=True, cwd=FD_DIR)

            # now build FD
            subprocess.run(["python", "build.py", "-j16"],
                           check=True,
                           cwd=FD_DIR)


def get_plan_file_path(root_dir, base_name):
    """Get path to best plan file from an FD planner. Generally the plan file
    path will be either `root_dir + '/' + basename` (for non-anytime planners)
    or `root_dir + '/' + basename + '.' + str(stage_num)` for anytime planners.
    In the latter case we want to get the plan file with the highest
    stage_num!"""
    file_names = os.listdir(root_dir)
    if base_name in file_names:
        return osp.join(root_dir, base_name)
    names_nums = []
    for file_name in file_names:
        if file_name.startswith(base_name):
            name_str, num_str = file_name.rsplit('.', 1)
            # the name should be something like `<basename>.3` (for third
            # stage, in this case)
            assert name_str == base_name, "can't handle name '%s' in '%s'" \
                % (file_name, root_dir)
            file_num = int(num_str)
            names_nums.append((file_num, file_name))
    _, latest_plan = max(names_nums)
    return osp.join(root_dir, latest_plan)


def _make_problem_txt(planner_exts):
    # returns domain_as_text, domain_name, problem_as_text, problem_name
    return extract_domain_problem(planner_exts.pddl_files,
                                  planner_exts.problem_name)


class FDTimeout(Exception):
    """Exception class for when Fast Downward times out on a problem."""
    pass


# regex to scrape the domain or problem names out of a (P)PDDL file
_NAME_RE = re.compile(
    r'^\s*\(\s*define\s+\((?:domain|problem)\s+([^\s\)]+)\s*\)')


def _name_from_txt(pddl_txt):
    """Takes a PDDL declaration & figures out name of the corresponding
    domain/problem. Assumes no comments & only one domain/problem in the
    string!"""
    # (starting with a very simple solution; if this doesn't work, then I'll
    # have to try running pddl_txt through the functions for
    # parsing/stringifying in pddl_utils.py)
    name, = _NAME_RE.findall(pddl_txt)
    return name


def run_fd_raw(planner,
               domain_txt,
               problem_txt,
               result_dir,
               *,
               timeout_s=None,
               cost_bound=None,
               mem_limit_mb=None):
    """Runs FD in a given directory & then returns path to directory."""

    assert has_fd(), \
        "Couldn't find Fast Downward. Use try_install_fd() before using this."

    # now setup output dir & write problem text
    os.makedirs(result_dir, exist_ok=True)
    domain_bn = 'domain.pddl'
    problem_bn = 'problem.pddl'
    domain_path = osp.join(result_dir, domain_bn)
    problem_path = osp.join(result_dir, problem_bn)
    with open(domain_path, 'w') as dom_fp, open(problem_path, 'w') as prob_fp:
        dom_fp.write(domain_txt)
        prob_fp.write(problem_txt)

    cost_bound_s = 'infinity' if cost_bound is None else str(cost_bound)
    del cost_bound  # so that I don't accidentally use it

    # figure out where FD is and run it
    cmdline = ["python", FD_PATH]
    if timeout_s is not None:
        assert timeout_s >= 1, "can't have <1s time limit (got %s)" % timeout_s
        cmdline.extend(["--overall-time-limit", "%ds" % timeout_s])
    if mem_limit_mb is not None:
        assert mem_limit_mb >= 1, "can't have <1MB memory limit (got %s)" \
            % mem_limit_mb
        cmdline.extend(["--overall-memory-limit", "%dM" % mem_limit_mb])
    cmdline.extend(['--plan-file', PLAN_BN])

    def make_wlama_args(w):
        # flags for LAMA's WA* thing using a specific W (the last ~3 or so
        # stages of LAMA are like this)
        assert isinstance(w, int) and w > 0
        return [
            "--evaluator", "ff=ff()", "--evaluator",
            "hlm=lmcount(lm_rhw(reasonable_orders=true),pref=true)",
            "--search",
            "lazy_wastar([ff,hlm],preferred=[ff,hlm],w={w},bound={bound})".
            format(w=w, bound=cost_bound_s)
        ]

    # set planner flags appropriately
    if planner == 'lama-2011':
        # This is not quite real LAMA-2011 preset because it only supports unit
        # costs (like the ASNet trainer, which I also need to fix). It should
        # be sufficient for our benchmark problems though.
        cost_prefs_bound = "[hff,hlm],preferred=[hff,hlm],bound={bound}" \
            .format(bound=cost_bound_s)
        cmdline.extend([
            domain_bn, problem_bn, "--evaluator",
            "hlm=lmcount(lm_rhw(reasonable_orders=true),pref={pref})".format(
                pref=True), "--evaluator", "hff=ff()", "--search",
            ("iterated([lazy_greedy({cost_prefs}),"
             "lazy_wastar({cost_prefs},w=5),lazy_wastar({cost_prefs},w=3),"
             "lazy_wastar({cost_prefs},w=2),lazy_wastar({cost_prefs},w=1)],"
             "repeat_last=true,continue_on_fail=true,bound={bound})").format(
                 cost_prefs=cost_prefs_bound, bound=cost_bound_s)
        ])
    elif planner == 'lama-first':
        # LAMA-first:
        # fast-downward.py --alias lama-first ${dom} ${prob}
        cmdline.extend([
            domain_bn, problem_bn, "--evaluator",
            ("hlm=lmcount(lm_factory=lm_rhw(reasonable_orders=true),"
             "transform=adapt_costs(one),pref=false)"), "--evaluator",
            "hff=ff(transform=adapt_costs(one))", "--search",
            ("lazy_greedy([hff,hlm],preferred=[hff,hlm],cost_type=one,"
             "reopen_closed=false,bound={bound})").format(bound=cost_bound_s)
        ])
    elif planner == 'lama-w5':
        # the second (i.e fourth-last) stage of LAMA-2011 (lazy WA* with W=5)
        cmdline.extend([
            domain_bn,
            problem_bn,
            *make_wlama_args(5),
        ])
    elif planner == 'lama-w3':
        # the third (also third-last) stage of LAMA-2011 (lazy WA* with W=3)
        cmdline.extend([
            domain_bn,
            problem_bn,
            *make_wlama_args(3),
        ])
    elif planner == 'lama-w2':
        # the second-last stage of LAMA-2011 (lazy WA* with W=2)
        cmdline.extend([
            domain_bn,
            problem_bn,
            *make_wlama_args(2),
        ])
    elif planner == 'lama-w1':
        # One step of the last stage of LAMA-2011 (lazy A*, so W=1). Note that
        # there is no *real* "last stage" of LAMA, since it keeps applying the
        # final stage search engine with tighter & tighter upper bounds on cost
        # until it runs out of time or something like that (using iterated
        # search). That means that it ends up with much better solutions than
        # you can get in just one application of the planner!
        cmdline.extend([
            domain_bn,
            problem_bn,
            *make_wlama_args(1),
        ])
    elif planner == 'astar-lmcut':
        # A* with LM-cut:
        # fast-downward.py ${dom} ${prob} --search "astar(lmcut())"
        # (similar template for astar or gbf with other heuristics)
        cmdline.extend([
            domain_bn, problem_bn, "--search",
            "astar(lmcut(),bound={bound})".format(bound=cost_bound_s)
        ])
    elif planner == 'astar-lmcount':
        # inadmissible variant of above
        cmdline.extend([
            domain_bn, problem_bn, "--search",
            "astar(lmcount(lm_rhw()),bound={bound})".format(bound=cost_bound_s)
        ])
    elif planner == 'astar-hadd':
        cmdline.extend([
            domain_bn, problem_bn, "--search",
            "astar(add(),bound={bound})".format(bound=cost_bound_s)
        ])
    elif planner == 'gbf-lmcut':
        # gbf = greedy best first (if this works well then finding a
        # generalised policy may be trivial for ASNets, using only action
        # landmarks!)
        cmdline.extend([
            domain_bn, problem_bn, "--search",
            "eager(single(lmcut()),bound={bound})".format(bound=cost_bound_s)
        ])
    elif planner == 'gbf-hadd':
        cmdline.extend([
            domain_bn, problem_bn, "--search",
            "eager(single(add()),bound={bound})".format(bound=cost_bound_s)
        ])
    else:
        raise ValueError("Unknown planner '%s'" % planner)

    # write command line to a text file so we can play back later if this fails
    cmdline_path = osp.join(result_dir, CMDLINE_BN)
    with open(cmdline_path, 'w') as cmdline_fp:
        cmdline_fp.write('\n'.join(cmdline))
        cmdline_fp.write('\n')

    # run FD, writing stderr/stdout to appropriate files
    out_path = osp.join(result_dir, STDOUT_BN)
    err_path = osp.join(result_dir, STDERR_BN)
    with open(out_path, 'w') as out_fp, open(err_path, 'w') as err_fp:
        rv = subprocess.Popen(cmdline,
                              cwd=result_dir,
                              stdout=out_fp,
                              stderr=err_fp,
                              universal_newlines=True)
    rv.wait()

    return rv


def run_fd_or_timeout(planner, domain_txt, problem_txt, *, timeout_s=None):
    """Takes a planner name recognised by FD (e.g `"lama-2011"`), a path to a
    PDDL domain, and a path to a PDDL problem, then produces a plan as list of
    state strings (or None, in case of failure to reach goal)."""
    domain_name = _name_from_txt(domain_txt)
    problem_name = _name_from_txt(domain_txt)
    dname = '%s:%s:%s' % (planner, domain_name, problem_name)
    guid = uuid.uuid1().hex
    result_dir = osp.join('/tmp', 'fd-results-%s-%s' % (dname, guid))

    run_fd_raw(planner=planner,
               domain_txt=domain_txt,
               problem_txt=problem_txt,
               result_dir=result_dir,
               timeout_s=timeout_s)

    # could use rv.returncode to check success, but that is sometimes
    # nonzero even when the planner manages to find a rough plan :(
    with open(osp.join(result_dir, STDOUT_BN), 'r') as stdout_fp:
        out_text = stdout_fp.read()
    with open(osp.join(result_dir, STDERR_BN), 'r') as stderr_fp:
        err_text = stderr_fp.read()
    with open(osp.join(result_dir, CMDLINE_BN), 'r') as cmdline_fp:
        cmdline = cmdline_fp.read()

    # Check whether search was successful or not. If it was not, return None;
    # if it was, return plan (as list of string-formatted action names, minus
    # parens).
    if 'Search stopped without finding a solution.' in out_text:
        rv = None
    elif (("Driver aborting after translate" in out_text
           or "Driver aborting after search" in out_text)
          and "Solution found." not in out_text):
        # timeout
        raise FDTimeout("FD appears to have timed out during search")
    else:
        assert 'Solution found.' in out_text, \
            "Bad stdout for cmd %s, ret code %r. Here is stdout:\n\n%s\n\n" \
            "Here is stderr:\n\n%s\n\n" \
            % (cmdline, getattr(rv, 'returncode', None), out_text, err_text)
        plan_path = get_plan_file_path(result_dir, PLAN_BN)

        with open(plan_path, 'r') as out_plan_fp:
            plan = []
            for line in out_plan_fp:
                line = line.strip()
                if line.startswith(';'):
                    continue
                assert line.startswith('(') and line.endswith(')')
                # strip parens (so we can use act_ident_to_mdpsim_act during
                # playback of solution)
                action = line[1:-1]
                plan.append(action)
            rv = plan

    # clean up output dir (we don't want to clean up in case of exception,
    # since output dir can be helpful)
    shutil.rmtree(result_dir)

    return rv


def _simulate_plan(init_cstate, plan_strs, planner_exts):
    """Simulate a plan to obtain a sequence of states. Will include all states
    visited by the plan in the order the are encountered, including initial
    state and goal state. Only works for deterministic problems, obviously!"""
    cstates = [init_cstate]
    costs = []
    for action_str in plan_strs:
        this_state = cstates[-1]
        assert not this_state.is_terminal
        action_id \
            = planner_exts.problem_meta.act_unique_id_to_index(action_str)
        next_state, cost = sample_next_state(cstates[-1], action_id,
                                             planner_exts)
        costs.append(cost)
        cstates.append(next_state)

    assert cstates[-1].is_terminal

    return cstates, costs


class FDQValueCache(object):
    """Cache of Q-values computed by calling FD repeatedly for the same
    problem."""

    def __init__(self,
                 planner_exts,
                 *,
                 planner='astar-hadd',
                 timeout_s=1800):
        # maps each state to a value computed via FD (states are represented by
        # tuples of true prop names, in no-paren format)
        self.planner_exts = planner_exts
        self.state_value_cache = {}
        self.best_action_cache = {}
        pddl_files = planner_exts.pddl_files
        problem_name = planner_exts.problem_name
        domain_hlist, domain_name, problem_hlist, problem_name_pddl = \
            extract_domain_problem(pddl_files, problem_name)
        assert problem_name == problem_name_pddl, \
            "name mismatch ('%r' != '%r')" % (problem_name, problem_name_pddl)
        self._domain_source = hlist_to_sexprs(domain_hlist)
        self._problem_hlist = problem_hlist
        self._domain_name = domain_name
        self._problem_name = problem_name
        self._planner_name = planner
        self._timeout_s = timeout_s
        self._fd_blacklist = set()

    def _run_fd_with_blacklist(self, *args, **kwargs):
        ident_tup = (args, tuple(sorted(kwargs)))
        if ident_tup in self._fd_blacklist:
            raise FDTimeout("this state previously caused planner timeout")
        try:
            return run_fd_or_timeout(*args, **kwargs)
        except FDTimeout:
            self._fd_blacklist.add(ident_tup)
            raise

    def compute_state_value_action(self, cstate):
        """Compute state value under V* (assumes the given planner is optimal,
        which it possibly is not) and also the action recommended by FD (may be
        None if no plan available). Caller may want to handle FDTimeout."""
        tup_state = cstate.to_fd_proplist()
        if tup_state in self.state_value_cache:
            return self.state_value_cache[tup_state], \
                self.best_action_cache[tup_state]
        if cstate.is_terminal:
            cost = 0 if cstate.is_goal else None
            self.state_value_cache[tup_state] = cost
            self.best_action_cache[tup_state] = None
            return cost, None
        # *_source is a string containing PDDL, *_hlist is the AST for the PDDL
        problem_hlist = replace_init_state(self._problem_hlist, tup_state)
        problem_source = hlist_to_sexprs(problem_hlist)
        plan = self._run_fd_with_blacklist(self._planner_name,
                                           self._domain_source,
                                           problem_source,
                                           timeout_s=self._timeout_s)
        if plan is None:
            # couldn't find a plan
            self.state_value_cache[tup_state] = None
            self.best_action_cache[tup_state] = None
            return None, None
        # visit all states except the last
        visited_states, step_costs \
            = _simulate_plan(cstate, plan, self.planner_exts)
        costs_to_goal = np.cumsum(step_costs[::-1])[::-1]
        visited_states = visited_states[:-1]
        assert len(visited_states) == len(plan), \
            "%d visited states, but %d actions in plan" \
            % (len(visited_states), len(plan))
        states_acts_costs = zip(visited_states, plan, costs_to_goal)
        for this_cstate, this_act, cost_to_goal in states_acts_costs:
            this_tup_state = this_cstate.to_fd_proplist()
            if this_tup_state in self.state_value_cache:
                old_val = self.state_value_cache[this_tup_state]
                if cost_to_goal > old_val:
                    continue
            self.state_value_cache[this_tup_state] = cost_to_goal
            self.best_action_cache[this_tup_state] = this_act
        return self.state_value_cache[tup_state], \
            self.best_action_cache[this_tup_state]

    def compute_state_value(self, cstate):
        value, _ = self.compute_state_value_action(cstate)
        return value

    def compute_q_values(self, cstate, act_strs):
        """Compute Q-values for each action applicable in the current state.
        Caller may want to handle FDTimeout."""
        q_values = []
        for act_str in act_strs:
            # expecting mdpsim format (for whatever reason)
            assert act_str[0] == '(' and act_str[-1] == ')'
            act_strip = act_str[1:-1]
            action_id = self.planner_exts.problem_meta \
                .act_unique_id_to_index(act_strip)
            is_enabled = cstate.acts_enabled[action_id][1]
            if not is_enabled:
                q_values.append(None)
                continue
            next_state, cost = sample_next_state(cstate, action_id,
                                                 self.planner_exts)
            next_state_value = self.compute_state_value(next_state)
            if next_state_value is None:
                # this action leads to a dead end
                q_values.append(None)
            else:
                q_value = cost + next_state_value
                q_values.append(q_value)
        return q_values

    def compute_policy_envelope(self, cstate):
        """Compute the 'optimal policy envelope' for a given `Canonical State`.
        Here `optimal policy envelope` really just means 'sequence of
        non-terminal states visited on the way to the goal'; this is a simpler
        notion than the notion of 'policy envelope' for a probabilistic
        problem. Caller may want to handle FDTimeout."""
        # warm up the value cache
        state_value = self.compute_state_value(cstate)
        if state_value is None:
            return []
        envelope = []
        while not cstate.is_terminal:
            envelope.append(cstate)
            best_act_str = self.best_action_cache[cstate.to_fd_proplist()]
            action_id = self.planner_exts.problem_meta \
                .act_unique_id_to_index(best_act_str)
            cstate, _ = sample_next_state(cstate, action_id, self.planner_exts)
            envelope.append(cstate)
            assert len(envelope) < 10000, \
                "envelope way too big, is there an infinite loop here?"
        return envelope


@contextlib.contextmanager
def _timer(task_name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print('[timer] %s took %fs' % (task_name, elapsed))


def _demo_main():
    # quick demo of what this file does
    # (exists in part to test its functionality)
    from asnets.supervised import PlannerExtensions

    # install FD first
    try_install_fd()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pddl_files',
        nargs='+',
        help='path to relevant PDDL files (domain, problem, etc.)')
    args = parser.parse_args()
    problem_name_re = re.compile(
        r'^\s*\(\s*define\s+\(problem\s+([^\s\)]+)\s*\)')
    for file_path in args.pddl_files[::-1]:
        with open(file_path, 'r') as fp:
            contents = fp.read()
        matches = problem_name_re.findall(contents)
        if matches:
            problem_name = matches[0]
            break
    else:
        raise ValueError("Could not find problem name in PDDL files: %s" %
                         ', '.join(args.pddl_files))
    planner_exts = PlannerExtensions(args.pddl_files, problem_name)
    value_cache = FDQValueCache(planner_exts)
    init_state = get_init_cstate(planner_exts)
    with _timer('Getting first set of values'):
        init_value = value_cache.compute_state_value(init_state)
    print('The value of the initial state is', init_value)
    all_act_strs = [
        '(%s)' % ba.unique_ident for ba, _ in init_state.acts_enabled
    ]
    enabled_act_strs = [
        '(%s)' % ba.unique_ident for ba, enabled in init_state.acts_enabled
        if enabled
    ]
    with _timer('Getting Q-values for %d actions' % len(enabled_act_strs)):
        q_values = value_cache.compute_q_values(init_state, enabled_act_strs)
    print('Q-values for initial state are', q_values)
    q_values_all = value_cache.compute_q_values(init_state, all_act_strs)
    print('Q-values for *all* actions are', q_values_all)
    policy_envelope = value_cache.compute_policy_envelope(init_state)
    print('FD policy envelope for initial state is', [
        ' '.join('(%s)' % p for p in s.to_fd_proplist())
        for s in policy_envelope
    ])


if __name__ == '__main__':
    _demo_main()
