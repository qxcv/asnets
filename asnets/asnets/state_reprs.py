"""Interface code for rllab. Mainly handles interaction with mdpsim & hard
things like action masking."""

from warnings import warn

import numpy as np

from asnets.prob_dom_meta import BoundProp, BoundAction
from asnets.py_utils import strip_parens


class CanonicalState(object):
    """The ASNet code uses a lot of state representations. There are
    pure-Python state representations, there are state representations based on
    the SSiPP & MDPsim wrappers, and there are string-based intermediate
    representations used to convert between the other representations. This
    class aims to be a single canonical state class that it is:

    1. Possible to convert to any other representation,
    2. Possible to instantiate from any other representation,
    3. Possible to pickle & send between processes.
    4. Efficient to manipulate.
    5. Relatively light on memory."""

    def __init__(self,
                 bound_prop_truth,
                 bound_acts_enabled,
                 is_goal,
                 *,
                 data_gens=None,
                 prev_cstate=None,
                 prev_act=None,
                 is_init_cstate=None):
        # note: props and acts should always use the same order! I don't want
        # to be passing around extra data to store "real" order for
        # propositions and actions all the time :(
        # FIXME: replace props_true and acts_enabled with numpy ndarray masks
        # instead of inefficient list-of-tuples structure
        self.props_true = tuple(bound_prop_truth)
        self.acts_enabled = tuple(bound_acts_enabled)
        self.is_goal = is_goal
        self.is_terminal = is_goal or not any(
            enabled for _, enabled in self.acts_enabled)
        self._aux_data = None
        self._aux_data_interp = None
        self._aux_data_interp_to_id = None
        if data_gens is not None:
            self.populate_aux_data(data_gens,
                                   prev_cstate=prev_cstate,
                                   prev_act=prev_act,
                                   is_init_cstate=is_init_cstate)
        # FIXME: make _do_validate conditional on a debug flag or something (I
        # suspect it's actually a bit expensive, so not turning on by default)
        # self._do_validate()

    def _do_validate(self):
        # this function runs some sanity checks on the newly-constructed state

        # first check proposition mask
        for prop_idx, prop_tup in enumerate(self.props_true):
            # should be tuple of (proposition, truth value)
            assert isinstance(prop_tup, tuple) and len(prop_tup) == 2
            assert isinstance(prop_tup[0], BoundProp)
            assert isinstance(prop_tup[1], bool)
            if prop_idx > 0:
                # should come after previous proposition alphabetically
                assert prop_tup[0].unique_ident \
                    > self.props_true[prop_idx - 1][0].unique_ident

        # next check action mask
        for act_idx, act_tup in enumerate(self.acts_enabled):
            # should be tuple of (action, enabled flag)
            assert isinstance(act_tup, tuple) and len(act_tup) == 2
            assert isinstance(act_tup[0], BoundAction)
            assert isinstance(act_tup[1], bool)
            if act_idx > 0:
                # should come after previous action alphabetically
                assert act_tup[0].unique_ident \
                    > self.acts_enabled[act_idx - 1][0].unique_ident

        # make sure that auxiliary data is 1D ndarray
        if self._aux_data is not None:
            assert isinstance(self._aux_data, np.ndarray), \
                "_aux_data is not ndarray (%r)" % type(self._aux_data)
            assert self._aux_data.ndim == 1

    def __repr__(self):
        # Python-legible state
        return '%s(%r, %r)' \
            % (self.__class__.__name__, self.props_true, self.acts_enabled)

    def __str__(self):
        # human-readable state
        prop_str = ', '.join(p.unique_ident for p, t in self.props_true if t)
        prop_str = prop_str or '-'
        extras = [
            'has aux_data' if self._aux_data is not None else 'no aux_data'
        ]
        if self.is_goal:
            extras.append('is goal')
        if self.is_terminal:
            extras.append('is terminal')
        state_str = 'State %s (%s)' % (prop_str, ', '.join(extras))
        return state_str

    def _ident_tup(self):
        # This function is used to get a hashable representation for __hash__
        # and __eq__. Note that we don't hash _aux_data because it's an
        # ndarray; instead, hash bool indicating whether we have _aux_data.
        # Later on, we WILL compare on _aux_data in the __eq__ method.
        # (probably it's a bad idea not to include that in the hash, but
        # whatever)
        return (self.props_true, self.acts_enabled, self._aux_data is None)

    def __hash__(self):
        return hash(self._ident_tup())

    def __eq__(self, other):
        if not isinstance(other, CanonicalState):
            raise TypeError(
                "Can't compare self (type %s) against other object (type %s)" %
                (type(self), type(other)))
        eq_basic = self._ident_tup() == other._ident_tup()
        if self._aux_data is not None and eq_basic:
            # equality depends on _aux_data being similar/identical
            return np.allclose(self._aux_data, other._aux_data)
        return eq_basic

    ##################################################################
    # Functions for dealing with ActionDataGenerators
    ##################################################################

    @property
    def aux_data(self):
        """Get auxiliary data produced by data generators."""
        if self._aux_data is None:
            raise ValueError("Must run .populate_aux_data() on state before "
                             "using .aux_data attribute.")
        return self._aux_data

    def populate_aux_data(self,
                          data_gens,
                          *,
                          prev_cstate=None,
                          prev_act=None,
                          is_init_cstate=None):
        """Populate class with auxiliary data from data generators."""
        extra_data = []
        interp = []
        requires_memory = False
        for dg in data_gens:
            dg_data = dg.get_extra_data(self,
                                        prev_cstate=prev_cstate,
                                        prev_act=prev_act,
                                        is_init_cstate=is_init_cstate)
            extra_data.append(dg_data)
            interp.extend(dg.dim_names)
            requires_memory |= dg.requires_memory
        if len(extra_data) == 0:
            num_acts = len(self.acts_enabled)
            self._aux_data = np.zeros((num_acts, ), dtype='float32')
        else:
            self._aux_data = np.concatenate(
                extra_data, axis=1).astype('float32').flatten()
        self._aux_dat_interp = interp
        if requires_memory:
            # one of the memory-based DataGenerators (ActionCountDataGenerator)
            # needs to know what slots the dims map onto.
            self._aux_data_interp_to_id = {
                dim_name: idx for idx, dim_name in enumerate(interp)
            }

    ##################################################################
    # MDPSim interop routines
    ##################################################################

    @classmethod
    def from_mdpsim(cls,
                    mdpsim_state,
                    planner_exts,
                    *,
                    prev_cstate=None,
                    prev_act=None,
                    is_init_cstate=None):
        # general strategy: convert both props & actions to string repr, then
        # use those reprs to look up equivalent BoundProposition/BoundAction
        # representation from problem_meta
        data_gens = planner_exts.data_gens
        problem_meta = planner_exts.problem_meta
        mdpsim_props_true \
            = planner_exts.mdpsim_problem.prop_truth_mask(mdpsim_state)
        truth_val_by_name = {
            # <mdpsim_prop>.identifier includes parens around it, which we want
            # to strip
            strip_parens(mp.identifier): truth_value
            for mp, truth_value in mdpsim_props_true
        }
        # now build mask from actual BoundPropositions in right order
        prop_mask = [(bp, truth_val_by_name[bp.unique_ident])
                     for bp in problem_meta.bound_props_ordered]

        # similar stuff for action selection
        mdpsim_acts_enabled \
            = planner_exts.mdpsim_problem.act_applicable_mask(mdpsim_state)
        act_on_by_name = {
            strip_parens(ma.identifier): enabled
            for ma, enabled in mdpsim_acts_enabled
        }
        act_mask = [(ba, act_on_by_name[ba.unique_ident])
                    for ba in problem_meta.bound_acts_ordered]

        is_goal = mdpsim_state.goal()

        return cls(prop_mask,
                   act_mask,
                   is_goal,
                   data_gens=data_gens,
                   prev_cstate=prev_cstate,
                   prev_act=prev_act,
                   is_init_cstate=is_init_cstate)

    def _to_state_string(self):
        # convert this state to a SSiPP-style state string
        ssipp_string = ', '.join(bp.unique_ident
                                 for bp, is_true in self.props_true if is_true)
        # XXX: remove this check once method tested
        assert ')' not in ssipp_string and '(' not in ssipp_string
        return ssipp_string

    def to_mdpsim(self, planner_exts):
        # yes, for some reason I originally made MDPSim take SSiPP-style
        # strings in this *one* place
        ssipp_style_string = self._to_state_string()
        problem = planner_exts.mdpsim_problem
        mdpsim_state = problem.intermediate_atom_state(ssipp_style_string)
        return mdpsim_state

    ##################################################################
    # SSiPP interop routines
    ##################################################################

    @classmethod
    def from_ssipp(cls,
                   ssipp_state,
                   planner_exts,
                   *,
                   prev_cstate=None,
                   prev_act=None,
                   is_init_cstate=None):
        problem = planner_exts.ssipp_problem
        problem_meta = planner_exts.problem_meta
        data_gens = planner_exts.data_gens
        ssipp_string = problem.string_repr(ssipp_state)

        # I made the (poor) decision of having string_repr return a string of
        # the form "(foo bar baz) (spam ham)" rather than "foo bar baz, spam
        # ham", so I need to strip_parens() here too (still had the foresight
        # to make string_repr return statics though---great!)
        true_prop_names = {
            p
            for p in ssipp_string.strip('()').split(') (') if p
        }

        # sanity check our hacky parse job
        bp_name_set = set(x.unique_ident
                          for x in problem_meta.bound_props_ordered)
        assert set(true_prop_names) <= bp_name_set
        prop_mask = [(bp, bp.unique_ident in true_prop_names)
                     for bp in problem_meta.bound_props_ordered]
        assert len(true_prop_names) == sum(on for _, on in prop_mask)

        # actions are a little harder b/c of ABSTRACTIONS!
        ssp = planner_exts.ssipp_ssp_iface
        ssipp_on_acts = ssp.applicableActions(ssipp_state)
        # yup, need to strip parens from actions too
        ssipp_on_act_names = {strip_parens(a.name()) for a in ssipp_on_acts}
        # FIXME: this is actually a hotspot in problems with a moderate number
        # of actions (say 5k+) if we need to create and delete a lot of states.
        act_mask = [(ba, ba.unique_ident in ssipp_on_act_names)
                    for ba in problem_meta.bound_acts_ordered]
        assert len(ssipp_on_act_names) \
            == sum(enabled for _, enabled in act_mask)

        # finally get goal flag
        is_goal = ssp.isGoal(ssipp_state)

        return cls(prop_mask,
                   act_mask,
                   is_goal,
                   data_gens=data_gens,
                   prev_cstate=prev_cstate,
                   prev_act=prev_act,
                   is_init_cstate=is_init_cstate)

    def to_ssipp(self, planner_exts):
        ssipp_string = self._to_state_string()
        problem = planner_exts.ssipp_problem
        ssipp_state = problem.get_intermediate_state(ssipp_string)
        return ssipp_state

    ##################################################################
    # FD routines (for passing stuff to FD)
    ##################################################################

    def to_fd_proplist(self):
        # just returns tuple of names of true propositions
        return tuple(bp.unique_ident for bp, truth in self.props_true if truth)

    ##################################################################
    # Network input routines (prepares flat vector to give to ASNet)
    ##################################################################

    def to_network_input(self):
        # need this first; user should populate it by calling
        # populate_aux_data() before calling to_network_input()
        aux_data = self.aux_data
        # will be 1 for true props, 0 for false props
        props_conv = np.array([truth for prop, truth in self.props_true],
                              dtype='float32')
        # will be 1 for enabled actions, 0 for disabled actions
        act_mask_conv = np.array(
            [enabled for act, enabled in self.acts_enabled], dtype='float32')
        # see PropNetwork._make_network for network input specification
        to_concat = (act_mask_conv, aux_data, props_conv)
        rv = np.concatenate(to_concat)
        return rv


def get_init_cstate(planner_exts):
    mdpsim_init = planner_exts.mdpsim_problem.init_state()
    cstate_init = CanonicalState.from_mdpsim(mdpsim_init,
                                             planner_exts,
                                             prev_cstate=None,
                                             prev_act=None,
                                             is_init_cstate=True)
    return cstate_init


def sample_next_state(cstate, action_id, planner_exts):
    assert isinstance(action_id, int), \
        "Action must be integer, but is %s" % type(action_id)
    assert isinstance(cstate, CanonicalState)

    # TODO: instead of passing in action_id, pass in a BoundAction
    mdpsim_state = cstate.to_mdpsim(planner_exts)
    bound_act, applicable = cstate.acts_enabled[action_id]
    if not applicable:
        tot_enabled = sum(truth for _, truth in cstate.acts_enabled)
        tot_avail = len(cstate.acts_enabled)
        warn('Selected disabled action #%d, %s; %d/%d available' %
             (action_id, bound_act, tot_enabled, tot_avail))
        new_cstate = cstate
    else:
        act_ident = bound_act.unique_ident
        mdpsim_action = planner_exts.act_ident_to_mdpsim_act[act_ident]
        new_mdpsim_state = planner_exts.mdpsim_problem.apply(
            mdpsim_state, mdpsim_action)
        new_cstate = CanonicalState.from_mdpsim(new_mdpsim_state,
                                                planner_exts,
                                                prev_cstate=cstate,
                                                prev_act=bound_act,
                                                is_init_cstate=False)

    # XXX: I don't currently calculate the real cost; must sort that
    # out!
    step_cost = 1

    return new_cstate, step_cost


def successors(cstate, action_id, planner_exts):
    bound_act, applicable = cstate.acts_enabled[action_id]
    if not applicable:
        raise ValueError("Action #%d is not enabled (action: %s)" %
                         (action_id, bound_act))
    ssipp_state = cstate.to_ssipp(planner_exts)
    act_ident = bound_act.unique_ident
    ssipp_action = planner_exts.ssipp_problem.find_action("(%s)" % act_ident)
    cost = ssipp_action.cost(ssipp_state)
    assert cost == 1, \
        "I don't think rest of the code can deal with cost of %s" % (cost, )
    # gives us a list of (probability, ssipp successor state) tuples
    ssipp_successors = planner_exts.ssipp.successors(
        planner_exts.ssipp_ssp_iface, ssipp_state, ssipp_action)
    canon_successors = [(p,
                         CanonicalState.from_ssipp(s,
                                                   planner_exts,
                                                   prev_cstate=cstate,
                                                   prev_act=bound_act,
                                                   is_init_cstate=False))
                        for p, s in ssipp_successors]
    return canon_successors


def get_action_name(planner_exts, action_id):
    acts_ordered = planner_exts.problem_meta.bound_acts_ordered
    if 0 <= action_id < len(acts_ordered):
        bound_act = acts_ordered[action_id]
        return bound_act.unique_ident
    return None  # explicit return b/c silent failure is intentional


def compute_observation_dim(planner_exts):
    extra_dims = sum(dg.extra_dim for dg in planner_exts.data_gens)
    nprops = planner_exts.mdpsim_problem.num_props
    nacts = planner_exts.mdpsim_problem.num_actions
    return nprops + (1 + extra_dims) * nacts


def compute_action_dim(planner_exts):
    return planner_exts.mdpsim_problem.num_actions
