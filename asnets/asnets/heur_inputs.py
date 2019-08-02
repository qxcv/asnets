"""Classes that generate extra information to pass to an ASNet, beyond
just proposition truth values."""
import abc

import numpy as np

from asnets import ssipp_interface
from asnets.py_utils import weak_ref_to, strip_parens


class ActionDataGenerator(abc.ABC):
    """ABC for things which stuff extra data into the input of each action."""

    def get_extra_data(self,
                       cstate,
                       *,
                       prev_cstate=None,
                       prev_act=None,
                       is_init_cstate=False):
        if self.requires_memory:
            assert is_init_cstate is not None
            assert is_init_cstate or (prev_cstate is not None
                                      and prev_act is not None)
            return self.get_extra_data_with_memory(
                cstate,
                prev_cstate=prev_cstate,
                prev_act=prev_act,
                is_init_cstate=is_init_cstate)
        return self.get_extra_data_no_memory(cstate)

    def get_extra_data_no_memory(self, cstate):
        """Get extra data from a state (of type `CanonicalState`). Either this
        or `get_extra_data_with_memory()` must be implemented, depending on
        whether this class `.requires_history`."""
        raise NotImplementedError("should be implemented in subclass")

    def get_extra_data_with_memory(self, this_cstate, prev_cstate, prev_act,
                                   is_init_cstate):
        """Get extra data from a `CanonicalState`. Either this of
        `.get_extra_data_no_memory()` must be implemented (this is the one that
        gets implemented when `.requires_memory` is True)."""
        raise NotImplementedError("should be implemented in subclass")

    @abc.abstractproperty
    def extra_dim(self):
        """Elements of extra data added per action."""
        pass

    @abc.abstractproperty
    def dim_names(self):
        """Short, human-readable names for each extra dimension (e.g
        'is-goal'). Should be list of same length as `.extra_dim` containing
        string-valued names."""
        pass

    def is_dead_end(self, cstate):
        """Check whether this is a dead end (return False if unsure)."""
        return False

    @property
    def requires_memory(self):
        """Does this DataGenerator need history information?"""
        return False


class ActionEnabledGenerator(ActionDataGenerator):
    extra_dim = 1
    dim_names = ['is-enabled']

    def get_extra_data_no_memory(self, cstate):
        # vec of |A| elems with 1 for enabled action and 0 for disabled action
        out_vec = np.asarray([truth for _, truth in cstate.acts_enabled],
                             dtype='float32')
        # add trailing dim
        out_vec = out_vec[:, None]
        return out_vec


class SSiPPDataGenerator(ActionDataGenerator):
    """Basic class for generators which use SSiPP"""

    def __init__(self, mod_sandbox):
        # important to have only weak ref to sandbox because the sandbox also
        # has a ref to us (!)
        self.mod_sandbox = weak_ref_to(mod_sandbox)
        self.ssipp_problem = weak_ref_to(self.mod_sandbox.ssipp_problem)


class RelaxedDeadendDetector(SSiPPDataGenerator):
    """Checks for dead ends in delete relaxation. No extra_dim because it only
    looks for dead ends."""
    extra_dim = 0
    dim_names = []

    def __init__(self, mod_sandbox):
        super().__init__(mod_sandbox)
        self.evaluator = ssipp_interface.Evaluator(self.mod_sandbox, "h-max")

    def get_extra_data_no_memory(self, cstate):
        return np.zeros((len(cstate.acts_enabled), self.extra_dim))

    def is_dead_end(self, cstate):
        ssipp_state = cstate.to_ssipp(self.mod_sandbox)
        state_value = self.evaluator.eval_state(ssipp_state)
        is_dead_end = state_value >= self.mod_sandbox.ssipp_dead_end_value
        return is_dead_end


class LMCutDataGenerator(SSiPPDataGenerator):
    """Adds 'this is in a disjunctive cut'-type flags to propositions."""
    extra_dim = 3
    dim_names = ['in-any-cut', 'in-singleton-cut', 'in-last-cut']
    IN_ANY_CUT = 0
    IN_SINGLETON_CUT = 1
    # The last cut contains actions helpful actions at the current state, and
    # the first cut contains the final goal-achieving action. That's just a
    # convention in my SSiPP wrapper, of course.
    IN_LAST_CUT = 2

    def __init__(self, *args):
        super().__init__(*args)
        self.cutter = ssipp_interface.Cutter(self.mod_sandbox)

    def get_extra_data_no_memory(self, cstate):
        out_vec = np.zeros((len(cstate.acts_enabled), self.extra_dim))
        ssipp_state = cstate.to_ssipp(self.mod_sandbox)
        cuts = self.cutter.get_action_cuts(ssipp_state)
        in_unary_cut = set()
        in_any_cut = set()
        for cut in cuts:
            cut = frozenset(strip_parens(a) for a in cut)
            if len(cut) == 1:
                in_unary_cut.update(cut)
            if len(cut) >= 1:
                # all actions in cuts (unary cuts or not) end up here
                in_any_cut.update(cut)
        if cuts:
            in_last_cut = {strip_parens(a) for a in cuts[-1]}
        else:
            in_last_cut = set()
        all_act_names = [a.unique_ident for a, _ in cstate.acts_enabled]
        assert (in_unary_cut | in_any_cut) <= set(all_act_names), \
            "there are some things in cuts that aren't in action set"
        assert in_last_cut <= set(all_act_names), \
            "there are things in the last cut that aren't really actions (?!)"
        for idx, act_name in enumerate(all_act_names):
            if act_name in in_unary_cut:
                out_vec[idx][self.IN_SINGLETON_CUT] = 1
            if act_name in in_any_cut:
                out_vec[idx][self.IN_ANY_CUT] = 1
            if act_name in in_last_cut:
                out_vec[idx][self.IN_LAST_CUT] = 1
        return out_vec


class HeuristicDataGenerator(SSiPPDataGenerator):
    """Will create some trivial features based on successors yielded by an
    action. Concretely, produces an indicator vector with the following
    layout:
        0: action disabled?
        1: best outcome decreases heuristic?
        2: best outcome increases heuristic?
        3: best outcome keeps heuristic same?
    Can probably fiddle around with this quite a bit (e.g. to add delta in dead
    end probabilities, etc.).
    """
    # how many extra elements we add to each action info vector
    extra_dim = 4
    dim_names = [
        'heur-disabled', 'heur-decrease', 'heur-increase', 'heur-same'
    ]
    # indices of elements used to indicate certain action states
    DISABLED = 0
    DECREASE = 1
    INCREASE = 2
    SAME = 3

    def __init__(self, mod_sandbox, heuristic_name):
        super().__init__(mod_sandbox)
        self.heuristic_name = heuristic_name
        self.evaluator = ssipp_interface.Evaluator(self.mod_sandbox,
                                                   heuristic_name)

    def get_extra_data_no_memory(self, cstate):
        out_vec = np.zeros((len(cstate.acts_enabled), self.extra_dim))
        ssipp_state = cstate.to_ssipp(self.mod_sandbox)
        state_value = self.evaluator.eval_state(ssipp_state)
        for idx, act_tup in enumerate(cstate.acts_enabled):
            # FIXME: don't keep a second enable_set, that's silly---just
            # iterate over in "correct" order
            bound_act, enabled = act_tup
            act_name = bound_act.unique_ident
            if not enabled:
                out_vec[idx, self.DISABLED] = 1
            else:
                succ_probs_vals \
                    = self.evaluator.succ_probs_vals(ssipp_state, act_name)
                best_outcome = min(val for _, val in succ_probs_vals)
                if best_outcome < state_value:
                    out_vec[idx, self.DECREASE] = 1
                elif best_outcome == state_value:
                    out_vec[idx, self.SAME] = 1
                else:  # greater
                    assert best_outcome > state_value
                    out_vec[idx, self.INCREASE] = 1
        return out_vec

    def is_dead_end(self, cstate):
        ssipp_state = cstate.to_ssipp(self.mod_sandbox)
        state_value = self.evaluator.eval_state(ssipp_state)
        return state_value >= ssipp_interface.dead_end_value()


class ActionCountDataGenerator(ActionDataGenerator):
    """Counts number of times each action has been executed so far."""
    extra_dim = 1
    dim_name = 'action_count'
    dim_names = [dim_name]
    requires_memory = True

    def __init__(self, problem_meta):
        self.problem_meta = weak_ref_to(problem_meta)

    def get_extra_data_with_memory(self, this_cstate, prev_cstate, prev_act,
                                   is_init_cstate):
        if is_init_cstate:
            return np.zeros((len(this_cstate.acts_enabled), self.extra_dim))
        # FIXME: this is a really dumb way to do things; I should be keeping
        # track of this separately & passing it into ALL memory-based
        # functions. Also I should keep aux_data as 2D instead of 1D, since
        # that's less error-prone. Fix this if I ever need to do it again.
        extra_dim = max(prev_cstate._aux_data_interp_to_id.values()) + 1
        old_aux_reshaped = prev_cstate.aux_data.reshape((-1, extra_dim))
        prev_dim_id = prev_cstate._aux_data_interp_to_id[self.dim_name]
        # get previous count vector & increment relevant count by 1
        aux_data_1d = old_aux_reshaped[:, prev_dim_id].copy()
        act_id = self.problem_meta.act_unique_id_to_index(
            prev_act.unique_ident)
        aux_data_1d[act_id] += 1
        aux_data_2d = aux_data_1d.reshape((-1, 1))
        return aux_data_2d
