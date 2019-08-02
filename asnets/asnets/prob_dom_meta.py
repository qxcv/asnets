"""Stores metadata for problems and domains in a pure-Python format. This
information is theoretically all obtainable from MDPSim extension. However,
I've introduced these extra abstractions so that I can pickle that information
and pass it between processes. C++ extension data structures (including those
from the MDPSim extension) can't be easily pickled, so passing around
information taken *straight* from the extension would not work."""

from functools import lru_cache, total_ordering


class UnboundProp:
    """Represents a proposition which may have free parameters (e.g. as it will
    in an action). .bind() will ground it."""

    def __init__(self, pred_name, params):
        # TODO: what if some parameters are already bound? This might happen
        # when you have constants, for instance. Maybe cross that bridge when I
        # get to it.
        self.pred_name = pred_name
        self.params = tuple(params)

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.params)

    def __repr__(self):
        return 'UnboundProp(%r, %r)' % (self.pred_name, self.params)

    def bind(self, bindings):
        assert isinstance(bindings, dict), \
            "expected dict of named bindings"
        args = []
        for param_name in self.params:
            if param_name[0] != '?':
                # already bound to constant
                arg = param_name
            else:
                if param_name not in bindings:
                    raise ValueError(
                        "needed bind for parameter %s, didn't get one" %
                        param_name)
                arg = bindings[param_name]
            args.append(arg)
        return BoundProp(self.pred_name, args)

    def __eq__(self, other):
        if not isinstance(other, UnboundProp):
            return NotImplemented
        return self.pred_name == other.pred_name \
            and self.params == other.params

    def __hash__(self):
        return hash(self.pred_name) ^ hash(self.params)


@total_ordering
class BoundProp:
    """Represents a ground proposition."""

    def __init__(self, pred_name, arguments):
        self.pred_name = pred_name
        self.arguments = tuple(arguments)
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.arguments)

    def __repr__(self):
        return 'BoundProp(%r, %r)' % (self.pred_name, self.arguments)

    def _compute_unique_ident(self):
        # going to match SSiPP-style names (think "foo bar baz" rather than
        # sexpr-style "(foo bar baz)")
        unique_id = ' '.join((self.pred_name, ) + self.arguments)
        return unique_id

    def __eq__(self, other):
        if not isinstance(other, BoundProp):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other):
        if not isinstance(other, BoundProp):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self):
        return hash(self.unique_ident)


@total_ordering
class UnboundAction:
    """Represents an action that *may* be lifted. Use .bind() with an argument
    list to ground it."""

    def __init__(self, schema_name, param_names, rel_props):
        self.schema_name = schema_name
        self.param_names = tuple(param_names)
        self.rel_props = tuple(rel_props)

        assert isinstance(schema_name, str)
        assert all(isinstance(a, str) for a in self.param_names)
        assert all(isinstance(p, UnboundProp) for p in self.rel_props)

    def __repr__(self):
        return 'UnboundAction(%r, %r, %r)' \
            % (self.schema_name, self.param_names, self.rel_props)

    def _ident_tup(self):
        return (self.schema_name, self.param_names, self.rel_props)

    def __eq__(self, other):
        if not isinstance(other, UnboundAction):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __lt__(self, other):
        if not isinstance(other, UnboundAction):
            return NotImplemented
        # avoid using self.rel_props because that would need ordering on
        # UnboundProp instances
        return (self.schema_name, self.param_names) \
            < (other.schema_name, other.param_names)

    def __hash__(self):
        return hash(self._ident_tup())

    def bind(self, arguments):
        # arguments should be a list or tuple of strings
        if not isinstance(arguments, (list, str)):
            raise TypeError('expected args to be list or str')
        bindings = dict(zip(self.param_names, arguments))
        props = [prop.bind(bindings) for prop in self.rel_props]
        return BoundAction(self, arguments, props)

    def num_slots(self):
        # number of "slots" in the action
        return len(self.rel_props)


@total_ordering
class BoundAction:
    """Represents a ground action."""

    def __init__(self, prototype, arguments, props):
        self.prototype = prototype
        self.arguments = tuple(arguments)
        self.props = tuple(props)
        self.unique_ident = self._compute_unique_ident()

        assert isinstance(prototype, UnboundAction)
        assert all(isinstance(a, str) for a in self.arguments)
        assert all(isinstance(p, BoundProp) for p in self.props)

    def __repr__(self):
        return 'BoundAction(%r, %r, %r)' \
            % (self.prototype, self.arguments, self.props)

    def __str__(self):
        return 'Action %s(%s)' % (self.prototype.schema_name, ', '.join(
            self.arguments))

    def _compute_unique_ident(self):
        unique_id = ' '.join((self.prototype.schema_name, ) + self.arguments)
        return unique_id

    def __eq__(self, other):
        if not isinstance(other, BoundAction):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __lt__(self, other):
        if not isinstance(other, BoundAction):
            return NotImplemented
        return self.unique_ident < other.unique_ident

    def __hash__(self):
        return hash(self.unique_ident)

    def num_slots(self):
        return len(self.props)


class DomainMeta:
    def __init__(self, name, unbound_acts, pred_names):
        self.name = name
        self.unbound_acts = tuple(unbound_acts)
        self.pred_names = tuple(pred_names)

    def __repr__(self):
        return 'DomainMeta(%s, %s, %s)' \
            % (self.name, self.unbound_acts, self.pred_names)

    # rel_acts maps predicate name to names of relevant action schemas (no
    # dupes). Similarly, rel_pred_names maps action schema names to relevant
    # predicates (in order they appear; some dupes expected).
    @lru_cache(None)
    def rel_act_slots(self, predicate_name):
        assert isinstance(predicate_name, str)
        rv = []
        for ub_act in self.unbound_acts:
            act_rps = self.rel_pred_names(ub_act)
            for slot, other_predicate_name in enumerate(act_rps):
                if predicate_name != other_predicate_name:
                    continue
                rv.append((ub_act, slot))
            # FIXME: maybe the "slots" shouldn't be integers, but rather tuples
            # of names representing parameters of the predicate like in
            # commented code below? Could then make those names consistent with
            # naming of the unbound action's parameters.
            # for ub_prop in ub_act.rel_props:
            #     if ub_prop.pred_name != predicate_name:
            #         continue
            #     slot_ident = ub_prop.params
            #     rv.append((ub_act, slot_ident))
        return rv

    @lru_cache(None)
    def rel_pred_names(self, action):
        assert isinstance(action, UnboundAction)
        rv = []
        for unbound_prop in action.rel_props:
            # it's important that we include duplicates here!
            rv.append(unbound_prop.pred_name)
        return rv

    @property
    @lru_cache(None)
    def all_unbound_props(self):
        unbound_props = []
        ub_prop_set = set()
        ub_prop_dict = {}
        for unbound_act in self.unbound_acts:
            for ub_prop in unbound_act.rel_props:
                if ub_prop not in ub_prop_set:
                    unbound_props.append(ub_prop)
                    ub_prop_dict[ub_prop.pred_name] = ub_prop
                    # the set is just to stop double-counting
                    ub_prop_set.add(ub_prop)
        return unbound_props, ub_prop_dict

    def unbound_prop_by_name(self, predicate_name):
        _, ub_prop_dict = self.all_unbound_props
        return ub_prop_dict[predicate_name]

    def _ident_tup(self):
        return (self.name, self.unbound_acts, self.pred_names)

    def __eq__(self, other):
        if not isinstance(other, DomainMeta):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __hash__(self):
        return hash(self._ident_tup())


class ProblemMeta:
    # Some notes on members/properties
    # - pred_to_props is dict w/ predicate name => [proposition name]
    # - schema_to_acts is dict w/ lifted action name => [ground action name]
    # - prop_to_pred and act_to_schema basically invert those mappings (but
    #   they're many-to-one, so we don't need lists)
    # - rel_props is dict mapping ground action name => [relevant prop names]
    # - rel_acts is dict mapping prop name => [relevant ground action name]

    def __init__(self, name, domain, bound_acts_ordered, bound_props_ordered,
                 goal_props):
        self.name = name
        self.domain = domain
        self.bound_acts_ordered = tuple(bound_acts_ordered)
        self.bound_props_ordered = tuple(bound_props_ordered)
        self.goal_props = tuple(goal_props)

        self._unique_id_to_index = {
            bound_act.unique_ident: idx
            for idx, bound_act in enumerate(self.bound_acts_ordered)
        }

        # sanity checks
        assert set(self.goal_props) <= set(self.bound_props_ordered)

    def __repr__(self):
        return 'ProblemMeta(%s, %s, %s, %s, %s)' \
            % (self.name, self.domain, self.bound_acts_ordered,
               self.bound_props_ordered, self.goal_props)

    @property
    def num_props(self):
        return len(self.bound_props_ordered)

    @property
    def num_acts(self):
        return len(self.bound_acts_ordered)

    @lru_cache(None)
    def schema_to_acts(self, unbound_action):
        assert isinstance(unbound_action, UnboundAction)
        return [
            a for a in self.bound_acts_ordered if a.prototype == unbound_action
        ]

    @lru_cache(None)
    def pred_to_props(self, pred_name):
        assert isinstance(pred_name, str)
        return [
            p for p in self.bound_props_ordered if p.pred_name == pred_name
        ]

    def prop_to_pred(self, bound_prop):
        assert isinstance(bound_prop, BoundProp)
        return bound_prop.pred_name

    def act_to_schema(self, bound_act):
        assert isinstance(bound_act, BoundAction)
        return bound_act.prototype

    def rel_props(self, bound_act):
        assert isinstance(bound_act, BoundAction)
        # no need for special grouping like in rel_acts, since all props can be
        # concatenated before passing them in
        return bound_act.props

    @lru_cache(None)
    def rel_act_slots(self, bound_prop):
        assert isinstance(bound_prop, BoundProp)
        rv = []
        pred_name = self.prop_to_pred(bound_prop)
        for unbound_act, slot in self.domain.rel_act_slots(pred_name):
            bound_acts_for_schema = []
            for bound_act in self.schema_to_acts(unbound_act):
                if bound_prop == self.rel_props(bound_act)[slot]:
                    # TODO: is this the best way to do this? See comment in
                    # DomainMeta.rel_acts.
                    bound_acts_for_schema.append(bound_act)
            rv.append((unbound_act, slot, bound_acts_for_schema))
        # list of tuples, each of the form (unbound action, slot, list of
        # ground actions)
        return rv

    @lru_cache(None)
    def prop_to_pred_subtensor_ind(self, bound_prop):
        assert isinstance(bound_prop, BoundProp)
        pred_name = self.prop_to_pred(bound_prop)
        prop_vec = self.pred_to_props(pred_name)
        return prop_vec.index(bound_prop)

    @lru_cache(None)
    def act_to_schema_subtensor_ind(self, bound_act):
        assert isinstance(bound_act, BoundAction)
        unbound_act = self.act_to_schema(bound_act)
        schema_vec = self.schema_to_acts(unbound_act)
        return schema_vec.index(bound_act)

    @lru_cache(None)
    def _props_by_name(self):
        all_props = {
            prop.unique_ident: prop
            for prop in self.bound_props_ordered
        }
        return all_props

    @lru_cache(None)
    def bound_prop_by_name(self, string):
        all_props = self._props_by_name()
        return all_props[string]

    def act_unique_id_to_index(self, string):
        return self._unique_id_to_index[string]


def make_unbound_prop(mdpsim_lifted_prop):
    pred_name = mdpsim_lifted_prop.predicate.name
    terms = [t.name for t in mdpsim_lifted_prop.terms]
    return UnboundProp(pred_name, terms)


def make_bound_prop(mdpsim_ground_prop):
    pred_name = mdpsim_ground_prop.predicate.name
    arguments = []
    for term in mdpsim_ground_prop.terms:
        term_name = term.name
        arguments.append(term_name)
        # make sure it's really a binding
        assert not term.name.startswith('?'), \
            "term '%s' starts with '?'---sure it's not free?"
    bound_prop = BoundProp(pred_name, arguments)
    return bound_prop


def make_unbound_action(mdpsim_lifted_act):
    schema_name = mdpsim_lifted_act.name
    param_names = [
        param.name for param, _ in mdpsim_lifted_act.parameters_and_types
    ]
    rel_props = []
    rel_prop_set = set()
    for mdpsim_prop in mdpsim_lifted_act.involved_propositions:
        unbound_prop = make_unbound_prop(mdpsim_prop)
        if unbound_prop not in rel_prop_set:
            # ignore duplicates
            rel_prop_set.add(unbound_prop)
            rel_props.append(unbound_prop)
    return UnboundAction(schema_name, param_names, rel_props)


def make_bound_action(mdpsim_ground_act):
    lifted_act = make_unbound_action(mdpsim_ground_act.lifted_action)
    arguments = [arg.name for arg in mdpsim_ground_act.arguments]
    return lifted_act.bind(arguments)


def get_domain_meta(domain):
    """Extracts a nice, Pickle-able subset of the information contained in a
    domain so that we can construct the appropriate network weights."""
    pred_names = [p.name for p in domain.predicates]
    unbound_acts = map(make_unbound_action, domain.lifted_actions)
    return DomainMeta(domain.name, unbound_acts, pred_names)


def get_problem_meta(problem, domain_meta):
    """Like get_domain_meta, but for problems."""
    # we get given the real domain, but we also do a double-check to make sure
    # that it matches our problem
    other_domain = get_domain_meta(problem.domain)
    assert other_domain == domain_meta, \
        "%r\n!=\n%r" % (other_domain, domain_meta)

    # use network input orders implied by problem.propositions and
    # problem.ground_actions
    bound_props_ordered = []
    goal_props = []
    for mdpsim_prop in problem.propositions:
        bound_prop = make_bound_prop(mdpsim_prop)
        bound_props_ordered.append(bound_prop)
        if mdpsim_prop.in_goal:
            goal_props.append(bound_prop)
    # must sort these!
    bound_props_ordered.sort()

    prop_set = set(bound_props_ordered)
    ub_act_set = set(domain_meta.unbound_acts)
    bound_acts_ordered = []
    for mdpsim_act in problem.ground_actions:
        bound_act = make_bound_action(mdpsim_act)
        bound_acts_ordered.append(bound_act)

        # sanity  checks
        assert set(bound_act.props) <= prop_set, \
            "bound_act.props (for act %r) not inside prop_set; odd ones: %r" \
            % (bound_act.unique_ident, set(bound_act.props) - prop_set)
        assert bound_act.prototype in ub_act_set, \
            "%r (bound_act.prototype) is not in %r (ub_act_set)" \
            % (bound_act.protype, ub_act_set)
    # again, need to sort lexically
    bound_acts_ordered.sort()

    return ProblemMeta(problem.name, domain_meta, bound_acts_ordered,
                       bound_props_ordered, goal_props)
