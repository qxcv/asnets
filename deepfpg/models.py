"""Scalable planning controller architectures for RLLab."""

from functools import wraps, total_ordering, lru_cache
from warnings import warn
from typing import Any, Set, List, Dict, Iterable, Tuple  # noqa

import joblib
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core import layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import tensorflow as tf
import numpy as np

from tf_utils import masked_softmax

# rel_acts maps predicate name to names of relevant action schemas (no dupes).
# Similarly, rel_preds maps action schema names to relevant predicates (in
# order they appear; some dupes expected).
# DomainMeta = namedtuple('DomainMeta', […])
# - pred_to_props is dict w/ predicate name => [proposition name]
# - schema_to_acts is dict w/ lifted action name => [ground action name]
# - prop_to_pred and act_to_schema basically invert those mappings (but they're
#   many-to-one, so we don't need lists)
# - rel_props is dict mapping ground action name => [relevant prop names]
# - rel_acts is dict mapping prop name => [relevant ground action name]
# ProblemMeta = namedtuple('ProblemMeta', […])


def deprecated(help_msg):
    """Mark a function as deprecated, with help message to make users less
    confused :-)"""
    assert isinstance(help_msg, str), "message must be string"

    def takes_f(f):
        msg = '%s is deprecated; %s' % (f.__name__, help_msg)

        @wraps(f)
        def wraps_f(*args, **kwargs):
            warn(msg, DeprecationWarning)
            return f(*args, **kwargs)

        return wraps_f

    return takes_f


class DomainMeta:
    def __init__(self,
                 name: str,
                 unbound_acts: Iterable['UnboundAction'],
                 pred_names: Iterable[str]) -> None:
        self.name = name
        self.unbound_acts = tuple(unbound_acts)
        self.pred_names = tuple(pred_names)

    def __repr__(self) -> str:
        return 'DomainMeta(%s, %s, %s)' \
            % (self.name, self.unbound_acts, self.pred_names)

    @lru_cache(None)
    def rel_acts(self, predicate_name: str) -> List['UnboundAction']:
        assert isinstance(predicate_name, str)
        rv = []
        for act in self.unbound_acts:
            act_rps = self.rel_pred_names(act)
            if predicate_name in act_rps:
                # TODO: is ignoring really the right way to handle this? I
                # assume so, because the domin's rel_acts are really only used
                # to sort the problem's rel_acts, but I could be wrong.

                # ignore duplicates
                rv.append(act)
        return rv

    @lru_cache(None)
    def rel_pred_names(self, action: 'UnboundAction') -> List[str]:
        assert isinstance(action, UnboundAction)
        rv = []
        for unbound_prop in action.rel_props:
            rv.append(unbound_prop.pred_name)
        return rv

    def _ident_tup(self):
        return (self.name, self.unbound_acts, self.pred_names)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DomainMeta):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __hash__(self) -> int:
        return hash(self._ident_tup())


class ProblemMeta:
    def __init__(self,
                 name: str,
                 domain: DomainMeta,
                 bound_acts_ordered: Iterable['BoundAction'],
                 bound_props_ordered: Iterable['BoundProp'],
                 goal_props: Iterable['BoundProp']) -> None:
        self.name = name
        self.domain = domain
        self.bound_acts_ordered = tuple(bound_acts_ordered)
        self.bound_props_ordered = tuple(bound_props_ordered)
        self.goal_props = tuple(goal_props)

        # sanity checks
        assert set(self.goal_props) <= set(self.bound_props_ordered)

    def __repr__(self) -> str:
        return 'ProblemMeta(%s, %s, %s, %s, %s)' \
            % (self.name, self.domain, self.bound_acts_ordered,
               self.bound_props_ordered, self.goal_props)

    @property
    def num_props(self) -> int:
        return len(self.bound_props_ordered)

    @property
    def num_acts(self) -> int:
        return len(self.bound_acts_ordered)

    @lru_cache(None)
    def schema_to_acts(self, unbound_action: 'UnboundAction') \
            -> List['BoundAction']:
        assert isinstance(unbound_action, UnboundAction)
        return [
            a for a in self.bound_acts_ordered if a.prototype == unbound_action
        ]

    @lru_cache(None)
    def pred_to_props(self, pred_name: str) -> List['BoundProp']:
        assert isinstance(pred_name, str)
        return [
            p for p in self.bound_props_ordered if p.pred_name == pred_name
        ]

    def prop_to_pred(self, bound_prop: 'BoundProp') -> str:
        assert isinstance(bound_prop, BoundProp)
        return bound_prop.pred_name

    def act_to_schema(self, bound_act: 'BoundAction') -> 'UnboundAction':
        assert isinstance(bound_act, BoundAction)
        return bound_act.prototype

    def rel_props(self, bound_act: 'BoundAction') -> Tuple['BoundProp', ...]:
        assert isinstance(bound_act, BoundAction)
        # no need for special grouping like in rel_acts, since all props can be
        # concatenated before passing them in
        return bound_act.props

    @lru_cache(None)
    def rel_acts(self, bound_prop: 'BoundProp') \
            -> List[Tuple['UnboundAction', List['BoundAction']]]:
        assert isinstance(bound_prop, BoundProp)
        rv = []
        pred_name = self.prop_to_pred(bound_prop)
        for unbound_act in self.domain.rel_acts(pred_name):
            bound_acts_for_schema = []
            for bound_act in self.schema_to_acts(unbound_act):
                if bound_prop in self.rel_props(bound_act):
                    # TODO: is this the best way to do this? See comment in
                    # DomainMeta.rel_acts.
                    bound_acts_for_schema.append(bound_act)
            rv.append((unbound_act, bound_acts_for_schema))
        # list of lists of actions
        return rv

    @lru_cache(None)
    def prop_to_pred_subtensor_ind(self, bound_prop: 'BoundProp') -> int:
        assert isinstance(bound_prop, BoundProp)
        pred_name = self.prop_to_pred(bound_prop)
        prop_vec = self.pred_to_props(pred_name)
        return prop_vec.index(bound_prop)

    @lru_cache(None)
    def act_to_schema_subtensor_ind(self, bound_act: 'BoundAction') -> int:
        assert isinstance(bound_act, BoundAction)
        unbound_act = self.act_to_schema(bound_act)
        schema_vec = self.schema_to_acts(unbound_act)
        return schema_vec.index(bound_act)

    @lru_cache(None)
    def _acts_by_name(self):
        all_acts = {act.unique_ident: act for act in self.bound_acts_ordered}
        return all_acts

    @lru_cache(None)
    def bound_act_by_name(self, string: str) -> 'BoundAction':
        # turns something like `(name obj1 obj1)` into a `BoundAction`
        all_acts = self._acts_by_name()
        return all_acts[string]

    @lru_cache(None)
    def _props_by_name(self):
        all_props = {
            prop.unique_ident: prop
            for prop in self.bound_props_ordered
        }
        return all_props

    @lru_cache(None)
    def bound_prop_by_name(self, string: str) -> 'BoundProp':
        all_props = self._props_by_name()
        return all_props[string]

    @lru_cache(None)
    def act_to_output_ind(self, act: 'BoundAction') -> int:
        return self.bound_acts_ordered.index(act)


class UnboundProp:
    """Represents a proposition which may have free parameters (e.g. as it will
    in an action). .bind() will ground it."""

    def __init__(self, pred_name: str, params: Iterable[str]) -> None:
        # TODO: what if some parameters are already bound? This might happen
        # when you have constants, for instance. Maybe cross that bridge when I
        # get to it.
        self.pred_name = pred_name
        self.params = tuple(params)

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.params)

    def __repr__(self) -> str:
        return 'UnboundProp(%r, %r)' % (self.pred_name, self.params)

    def bind(self, bindings: Dict[str, str]) -> 'BoundProp':
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnboundProp):
            return NotImplemented
        return self.pred_name == other.pred_name \
            and self.params == other.params

    def __hash__(self) -> int:
        return hash(self.pred_name) ^ hash(self.params)


class BoundProp:
    """Represents a ground proposition."""

    def __init__(self, pred_name: str, arguments: Iterable[str]) -> None:
        self.pred_name = pred_name
        self.arguments = tuple(arguments)
        self.unique_ident = self._unique_ident()

        assert isinstance(self.pred_name, str)
        assert all(isinstance(p, str) for p in self.arguments)

    def __repr__(self) -> str:
        return 'BoundProp(%r, %r)' % (self.pred_name, self.arguments)

    def _unique_ident(self) -> str:
        # should match mdpsim-style names
        my_name = self.pred_name
        arg_strs = ''.join(' ' + arg for arg in self.arguments)
        return '(%s%s)' % (my_name, arg_strs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BoundProp):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __hash__(self) -> int:
        return hash(self.unique_ident)


@total_ordering
class UnboundAction:
    """Represents an action that *may* be lifted. Use .bind() with an argument
    list to ground it."""

    def __init__(self,
                 schema_name: str,
                 param_names: Iterable[str],
                 rel_props: Iterable[UnboundProp]) -> None:
        self.schema_name = schema_name
        self.param_names = tuple(param_names)
        self.rel_props = tuple(rel_props)

        assert isinstance(schema_name, str)
        assert all(isinstance(a, str) for a in self.param_names)
        assert all(isinstance(p, UnboundProp) for p in self.rel_props)

    def __repr__(self) -> str:
        return 'UnboundAction(%r, %r, %r)' \
            % (self.schema_name, self.param_names, self.rel_props)

    def _ident_tup(self):
        return (self.schema_name, self.param_names, self.rel_props)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, UnboundAction):
            return NotImplemented
        return self._ident_tup() == other._ident_tup()

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, UnboundAction):
            return NotImplemented
        # avoid using self.rel_props because that would need ordering on
        # UnboundProp instances
        return (self.schema_name, self.param_names) \
            < (other.schema_name, other.param_names)

    def __hash__(self) -> int:
        return hash(self._ident_tup())

    def bind(self, arguments):
        # arguments should be a list or tuple of strings
        if not isinstance(arguments, (list, str)):
            raise TypeError('expected args to be list or str')
        bindings = dict(zip(self.param_names, arguments))
        props = [prop.bind(bindings) for prop in self.rel_props]
        return BoundAction(self, arguments, props)


class BoundAction:
    """Represents a ground action."""

    def __init__(self,
                 prototype: UnboundAction,
                 arguments: Iterable[str],
                 props: Iterable[BoundProp]) -> None:
        self.prototype = prototype
        self.arguments = tuple(arguments)
        self.props = tuple(props)
        self.unique_ident = self._unique_ident()

        assert isinstance(prototype, UnboundAction)
        assert all(isinstance(a, str) for a in self.arguments)
        assert all(isinstance(p, BoundProp) for p in self.props)

    def __repr__(self) -> str:
        return 'BoundAction(%r, %r, %r)' \
            % (self.prototype, self.arguments, self.props)

    def _unique_ident(self) -> str:
        my_name = self.prototype.schema_name
        arg_strs = ''.join(' ' + arg for arg in self.arguments)
        return '(%s%s)' % (my_name, arg_strs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BoundAction):
            return NotImplemented
        return self.unique_ident == other.unique_ident

    def __hash__(self) -> int:
        return hash(self.unique_ident)


def make_unbound_prop(mdpsim_lifted_prop: Any) -> UnboundProp:
    pred_name = mdpsim_lifted_prop.predicate.name  # type: str
    terms = [t.name for t in mdpsim_lifted_prop.terms]  # type: List[str]
    return UnboundProp(pred_name, terms)


def make_bound_prop(mdpsim_ground_prop: Any) -> BoundProp:
    pred_name = mdpsim_ground_prop.predicate.name  # type: str
    arguments = []
    for term in mdpsim_ground_prop.terms:
        term_name = term.name  # type: str
        arguments.append(term_name)
        # make sure it's really a binding
        assert not term.name.startswith('?'), \
            "term '%s' starts with '?'---sure it's not free?"
    bound_prop = BoundProp(pred_name, arguments)
    return bound_prop


def make_unbound_action(mdpsim_lifted_act: Any) -> UnboundAction:
    schema_name = mdpsim_lifted_act.name  # type: str
    param_names = [
        param.name for param, _ in mdpsim_lifted_act.parameters_and_types
    ]  # type: List[str]
    rel_props = []
    rel_prop_set = set()  # type: Set[UnboundProp]
    for mdpsim_prop in mdpsim_lifted_act.involved_propositions:
        unbound_prop = make_unbound_prop(mdpsim_prop)
        if unbound_prop not in rel_prop_set:
            # ignore duplicates
            rel_prop_set.add(unbound_prop)
            rel_props.append(unbound_prop)
    return UnboundAction(schema_name, param_names, rel_props)


def make_bound_action(mdpsim_ground_act: Any) -> BoundAction:
    lifted_act = make_unbound_action(mdpsim_ground_act.lifted_action)
    arguments = [arg.name
                 for arg in mdpsim_ground_act.arguments]  # type: List[str]
    return lifted_act.bind(arguments)


def get_domain_meta(domain: Any) -> DomainMeta:
    """Extracts a nice, Pickle-able subset of the information contained in a
    domain so that we can construct the appropriate network weights."""
    pred_names = [p.name for p in domain.predicates]
    unbound_acts = map(make_unbound_action, domain.lifted_actions)
    return DomainMeta(domain.name, unbound_acts, pred_names)


def get_problem_meta(problem: Any, domain_meta: DomainMeta) -> ProblemMeta:
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

    return ProblemMeta(problem.name, domain_meta, bound_acts_ordered,
                       bound_props_ordered, goal_props)


class PropNetworkWeights:
    """Manages weights for a domain-specific problem network. Those weights can
    then be used in problem-specific networks."""

    # WARNING: you need to change __{get,set}state__ if you change __init__ or
    # _make_weights()!

    def __init__(self,
                 dom_meta: DomainMeta,
                 hidden_sizes: Iterable[Tuple[int, int]],
                 extra_dim: int=0) -> None:
        # note that hidden_sizes is list of (act layer size, prop layer size)
        # pairs.
        # extra_input is just the number of extra items included in the input
        # vector for each action
        self.dom_meta = dom_meta
        self.hidden_sizes = list(hidden_sizes)
        self.extra_dim = extra_dim
        self._make_weights()

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle weights ourselves, since TF stuff is hard to pickle."""
        prop_weights_np = self._serialise_weight_list(self.prop_weights)
        act_weights_np = self._serialise_weight_list(self.act_weights)
        return {
            'dom_meta': self.dom_meta,
            'hidden_sizes': self.hidden_sizes,
            'prop_weights_np': prop_weights_np,
            'act_weights_np': act_weights_np,
            'extra_dim': self.extra_dim
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickle weights"""
        self.dom_meta = state['dom_meta']
        self.hidden_sizes = state['hidden_sizes']
        self.extra_dim = state['extra_dim']
        self._make_weights(state['prop_weights_np'], state['act_weights_np'])

    @staticmethod
    def _serialise_weight_list(weight_list: List[Dict[str, Any]]) \
            -> List[Dict[str, Tuple[Any, ...]]]:
        # serialises a list of dicts, each mapping str -> (tensorflow weights,
        # ...)
        sess = tf.get_default_session()
        rv = []
        for d in weight_list:
            new_d = {}
            for k, v in d.items():
                new_d[k] = tuple(sess.run(v))
            rv.append(new_d)
        return rv

    def _make_weights(self, old_prop_weights=None, old_act_weights=None):
        # prop_weights[i] is a dictionary mapping predicate names to weights
        # for modules in the i-th proposition layer
        self.prop_weights = []
        self.act_weights = []
        self.all_weights = []

        for hid_idx, hid_sizes in enumerate(self.hidden_sizes):
            act_size, prop_size = hid_sizes

            # make action layer weights
            act_dict = {}
            for unbound_act in self.dom_meta.unbound_acts:
                preds = self.dom_meta.rel_pred_names(unbound_act)
                if not hid_idx:
                    # first layer, so our input is actually a binary vector
                    # giving a truth value for each proposition
                    in_size = len(preds) * 2 + self.extra_dim
                else:
                    in_size = len(preds) * self.hidden_sizes[hid_idx - 1][1]

                name_pfx = 'hid_%d_act_%s' % (hid_idx, unbound_act.schema_name)
                # TODO: doing "if old_act_weights" check each time is silly.
                # Should store parameters *purely* by name, and have code
                # responsible for automatically re-instantiating old weights if
                # they exist.
                if old_act_weights is not None:
                    W_init, b_init = map(L.const,
                                         old_act_weights[hid_idx][unbound_act])
                else:
                    W_init = L.XavierUniformInitializer()
                    b_init = tf.zeros_initializer()
                act_W = L.create_param(
                    W_init, shape=(1, in_size, act_size), name=name_pfx + '/W')
                act_b = L.create_param(
                    b_init, shape=(act_size, ), name=name_pfx + '/b')
                act_dict[unbound_act] = (act_W, act_b)
                self.all_weights.extend([act_W, act_b])

            self.act_weights.append(act_dict)

            # make hidden proposition layer weights
            pred_dict = {}
            for pred_name in self.dom_meta.pred_names:
                rel_acts = self.dom_meta.rel_acts(pred_name)
                # We should never end up with NO relevant actions for a
                # predicate. Why bother including the predicate?
                assert len(rel_acts) > 0, \
                    "no relevant actions for proposition %s" % pred_name

                in_size = len(rel_acts) * act_size
                name_pfx = 'hid_%d_prop_%s' % (hid_idx, pred_name)
                if old_prop_weights is not None:
                    W_init, b_init = map(L.const,
                                         old_prop_weights[hid_idx][pred_name])
                else:
                    W_init = L.XavierUniformInitializer()
                    b_init = tf.zeros_initializer()
                prop_W = L.create_param(
                    W_init,
                    shape=(1, in_size, prop_size),
                    name=name_pfx + '/W')
                prop_b = L.create_param(
                    b_init, shape=(prop_size, ), name=name_pfx + '/b')
                pred_dict[pred_name] = (prop_W, prop_b)
                self.all_weights.extend([prop_W, prop_b])

            self.prop_weights.append(pred_dict)

        # make final layer weights (action)
        final_act_dict = {}
        for unbound_act in self.dom_meta.unbound_acts:
            preds = self.dom_meta.rel_pred_names(unbound_act)
            if not self.hidden_sizes:
                in_size = len(preds) * 2 + self.extra_dim
            else:
                in_size = len(preds) * self.hidden_sizes[-1][1]

            name_pfx = 'final_act_%s' % unbound_act.schema_name
            if old_act_weights is not None:
                W_init, b_init = map(L.const, old_act_weights[-1][unbound_act])
            else:
                W_init = L.XavierUniformInitializer()
                b_init = tf.zeros_initializer()
            final_act_W = L.create_param(
                W_init, shape=(1, in_size, 1), name=name_pfx + '/W')
            final_act_b = L.create_param(
                b_init, shape=(1, ), name=name_pfx + '/b')
            final_act_dict[unbound_act] = (final_act_W, final_act_b)
            self.all_weights.extend([final_act_W, final_act_b])

        self.act_weights.append(final_act_dict)

    def save(self, path: str) -> None:
        """Save a snapshot of the current network weights to the given path."""
        joblib.dump(self, path, compress=True)


class ResponseNormLayer(L.Layer):
    """Somewhat normalise feature maps of a 1D conv filter."""

    def __init__(
            self,
            incoming,
            # adapted to be in same range as AlexNet paper (except alpha,
            # which is bigger because this layer divides by N)
            axis=1,
            kappa=2,
            alpha=0.02,
            beta=0.75,
            **kwargs):
        super(ResponseNormLayer, self).__init__(incoming, **kwargs)
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        in_size = tf.shape(input)[self.axis]
        squares = tf.square(input) / tf.cast(in_size, tf.float32)
        denom = (self.kappa + self.alpha * squares)**self.beta
        normed = input / denom
        return normed


class PickPoolAndStack(L.MergeLayer):
    """Takes some previous layers and some indices, and creates a new data
    layer by picking out the appropriate items then concatenating and/ or
    pooling them together.

    Yes, that description is extremely vague. In my defense, this layer is an
    important part of my implementation strategy for action/proposition
    modules, so it has to do a lot of unrelated things. Hopefully I'll be able
    to write a better explanation later."""

    def __init__(self, incomings, index_spec, **kwargs):
        super().__init__(incomings, **kwargs)
        assert len(index_spec) > 0, "need at least one incoming"
        self.index_spec = index_spec
        self._check_index_spec()

    def _check_index_spec(self) -> None:
        # shape of index_spec: [(int, [[int]])]
        # each element of index_spec is pair of (index into inputs, [[indices
        # into given input]]) Note that I have a nested list ([[int]]) because
        # each output could be *pooled* across multiple inputs.
        out_len = None

        for chosen_input, pools in self.index_spec:
            assert isinstance(chosen_input, int), \
                "input selector must be integer"

            for pool in pools:
                # I had a length check in before, but then realised that pools
                # could be empty :/
                # assert len(pool) >= 0
                assert all(isinstance(idx, int) for idx in pool), \
                    "indices in pool must be integers"

            if out_len is None:
                out_len = len(pools)
            else:
                assert out_len == len(pools), \
                    "need same pool count from each input, but %d != %d" \
                    % (out_len, len(pools))

    def get_output_shape_for(self, input_shapes: List[List[int]]) -> List[int]:
        # input is N * D * L (D is num input channels, L is length---hopefully)
        batch_size = None
        chans_out = 0
        out_len = None

        for chosen_input, pools in self.index_spec:
            this_shape = input_shapes[chosen_input]
            if batch_size is None:
                batch_size, _, D = this_shape
            else:
                batch_size_p, _, D = this_shape
                assert batch_size == batch_size_p, "Expected " \
                    "batch size %s, got %s" % (batch_size, batch_size_p)
            chans_out += D
            out_len = len(pools)

        return [batch_size, out_len, chans_out]

    def get_output_for(self, inputs, **kwargs):
        # out_stacks will be an array of len(self.index_spec) tensors, each N *
        # L * D_i (where sum of D_is is the chans_out value returned by
        # get_output_shape_for)
        out_stacks = []
        for chosen_input, pools in self.index_spec:
            this_input = inputs[chosen_input]
            this_in_shape = tf.shape(this_input)
            in_bs = this_in_shape[0]

            if all(len(p) == 1 for p in pools):
                # This is the fast path: we don't have to pool at all, because
                # there's precisely one element in each pool. This always
                # happens when constructing action modules, which never need
                # any sort of pooling.
                np_indices = np.array([idx for idx, in pools], dtype=np.int32)
                assert np_indices.ndim == 1, np_indices.shape
                this_stack = tf.gather(this_input, np_indices, axis=1)
                out_stacks.append(this_stack)
            else:
                # This is the slow path: we need to gather along a separate
                # axis, then pool along that axis.
                pool_sizes = np.array(list(map(len, pools)))
                max_size = pool_sizes.max()
                num_pools = len(pool_sizes)
                normed_pools = np.zeros((num_pools, max_size), dtype=np.int32)
                is_valid = np.zeros((num_pools, max_size), dtype=np.float32)
                for pool_idx, pool in enumerate(pools):
                    # pad end of array for gather() with zeros
                    normed_pools[pool_idx, :len(pool)] = pool
                    is_valid[pool_idx, :len(pool)] = 1.0
                gathered = tf.gather(this_input, normed_pools, axis=1)
                # mask out tensors which should not contribute to output
                masked = gathered * is_valid[None, :, :, None]
                # max pool
                pooled = tf.reduce_max(masked, axis=2)
                # this was for mean pooling (not as good)
                # summed = tf.reduce_sum(masked, axis=2)
                # divide through by actual pool sizes (where nonzero)
                # shaped_pool_sizes = pool_sizes[None, :, None]
                # safe_pool_sizes = tf.maximum(shaped_pool_sizes,
                #                              tf.ones_like(shaped_pool_sizes))
                # pooled = summed / tf.cast(safe_pool_sizes, tf.float32)
                out_stacks.append(pooled)

        # concat along input channels
        rv = tf.concat(out_stacks, axis=2)
        check_bs = tf.assert_equal(
            tf.shape(rv)[0],
            in_bs,
            message="returned batch size doesn't "
            "match that of last input")
        check_rank = tf.assert_rank(rv, 3, message='return rank is not 3')
        with tf.control_dependencies([check_bs, check_rank]):
            return tf.identity(rv)


class PropNetwork(LayersPowered, Serializable):
    """Concrete implementation of a proposition/action network """

    def __init__(self,
                 weight_manager: PropNetworkWeights,
                 problem_meta: ProblemMeta,
                 dropout: float=0.0,
                 norm_response: bool=False) -> None:
        Serializable.quick_init(self, locals())

        self._weight_manager = weight_manager
        self._prob_meta = problem_meta

        # ReLU dies before convergence on TTW-1
        # self.nonlinearity = tf.nn.relu

        # tanh learns better, but didn't fully converge on TTW-1 w/ one layer
        # and saturated on deeper networks
        # self.nonlinearity = tf.nn.tanh

        # softplus learns even better on TTW-1 w/ more layers, but results in
        # HUGE activations eventually, and ultimately NaNs (only after solving
        # the problem fully, though)
        # self.nonlinearity = tf.nn.softplus

        # Leaky ReLU doesn't make activations explode, and does seem to result
        # in some change in activation distribution. It doesn't learn quickly,
        # however, and seems to suffer from vanishing gradients in earlier
        # layers. Vanishing problem resolved after 60+ iterations, but gave way
        # to exploding activation problem (at least when pooling with
        # reduce_sum).
        # self.nonlinearity = make_leaky_relu(0.1)

        # elu takes a bit longer to converge than softplus (in wall time), but
        # a bit less than the leaky relu above. Seems to lead to even less
        # activation-explosion.
        self.nonlinearity = tf.nn.elu

        self.dropout = dropout
        self.norm_response = norm_response

        self._make_mlp()

        LayersPowered.__init__(self, [self._l_out])

    def _make_mlp(self):
        hidden_sizes = self._weight_manager.hidden_sizes
        dom_meta = self._weight_manager.dom_meta
        prob_meta = self._prob_meta

        # input vector spec:
        #
        # |<--num_acts-->|<--k*num_acts-->|<--num_props-->|
        # | action mask  |  action data   | propositions  |
        #
        # 1) `action_mask` tells us whether actions are enabled
        # 2) `action_data` is passed straight to action modules
        # 3) `propositions` tells us what is and isn't true
        #
        # Reminder: this convoluted input shape is required solely because of
        # rllab inflexible input conventions (it can only take a single vector
        # per state).

        mask_size = prob_meta.num_acts
        extra_data_dim = self._weight_manager.extra_dim
        extra_size = extra_data_dim * prob_meta.num_acts
        prop_size = prob_meta.num_props
        in_dim = mask_size + extra_size + prop_size
        l_in = L.InputLayer(shape=(None, in_dim))
        l_mask = L.OpLayer(
            l_in,
            lambda inv: inv[:, :mask_size],
            lambda s: s[:1] + (mask_size, ) + s[2:],
            name='split/mask')

        def act_extra_inner(in_vec):
            act_vecs = in_vec[:, mask_size:mask_size + extra_size]
            # unflatten
            # inner_shape = tf.TensorShape(
            #     (prob_meta.num_acts, extra_data_dim))
            # out_shape = act_vecs.shape[:1] + inner_shape
            out_shape = (-1, prob_meta.num_acts, extra_data_dim)
            return tf.reshape(act_vecs, out_shape)

        def obs_inner(in_vec):
            prop_truth = in_vec[:, mask_size + extra_size:, None]
            goal_vec = [
                float(prop in prob_meta.goal_props)
                for prop in prob_meta.bound_props_ordered
            ]

            assert sum(goal_vec) == len(prob_meta.goal_props)
            assert any(goal_vec), 'there are no goals?!'
            assert not all(goal_vec), 'there are no goals?!'

            # apparently this broadcasts (hooray!)
            tf_goals = tf.constant(goal_vec)[None, :, None]
            batch_size = tf.shape(prop_truth)[0]
            tf_goals_broad = tf.tile(tf_goals, (batch_size, 1, 1))
            return tf.concat([prop_truth, tf_goals_broad], axis=2)

        l_obs = L.OpLayer(
            l_in,
            obs_inner,
            lambda s: s[:1] + (prop_size, 2),
            name='split/obs')
        pred_dict = self._split_input(l_obs)
        if extra_data_dim > 0:
            l_act_extra = L.OpLayer(
                l_in,
                act_extra_inner,
                lambda s: s[:1] + (prob_meta.num_acts, extra_data_dim),
                name='split/extra')
            extra_dict = self._split_extra(l_act_extra)
        else:
            extra_dict = None

        # hidden layers
        for hid_idx, hid_sizes in enumerate(hidden_sizes):
            act_size, prop_size = hid_sizes

            act_dict = {}
            for unbound_act in dom_meta.unbound_acts:
                act_dict[unbound_act] = self._make_action_module(
                    pred_dict,
                    unbound_act,
                    act_size,
                    hid_idx,
                    l_in,
                    dropout=self.dropout,
                    norm_response=self.norm_response,
                    extra_dict=extra_dict)

            pred_dict = {}
            for pred_name in dom_meta.pred_names:
                pred_dict[pred_name] = self._make_prop_module(
                    act_dict,
                    pred_name,
                    prop_size,
                    hid_idx,
                    l_in,
                    dropout=self.dropout,
                    norm_response=self.norm_response)

        # final (action) layer
        finals = {}
        for unbound_act in dom_meta.unbound_acts:
            finals[unbound_act] = self._make_action_module(
                pred_dict,
                unbound_act,
                1,
                len(hidden_sizes),
                l_in,
                nonlinearity=tf.identity,
                # can't have ANY dropout in final layer!
                dropout=0.0,
                # or normalisation
                norm_response=False,
                extra_dict=extra_dict)
        l_pre_softmax = self._merge_finals(finals)
        self._l_out = L.OpLayer(
            l_pre_softmax, masked_softmax, extras=[l_mask], name='l_out')
        self._l_in = l_in

    def _split_input(self, obs_layer):
        """Splits an observation layer up into appropriate proposition
        layers."""
        # TODO: comment below about _merge_finals ugliness also applies here
        prob_meta = self._prob_meta
        prop_to_flat_input_idx = {
            prop: idx
            for idx, prop in enumerate(prob_meta.bound_props_ordered)
        }
        rv = {}

        for pred_name in prob_meta.domain.pred_names:
            sub_props = prob_meta.pred_to_props(pred_name)
            gather_inds = []
            for sub_prop_idx, sub_prop in enumerate(sub_props):
                to_look_up = prop_to_flat_input_idx[sub_prop]
                gather_inds.append(to_look_up)

            # closure circumlocutions because we're in a loop :(
            def make_gather_layer(inds_to_fetch, pred_name):
                return L.OpLayer(
                    obs_layer,
                    lambda v: tf.gather(v, inds_to_fetch, axis=1),
                    lambda s: s[:1] + (len(inds_to_fetch), s[1]),
                    name='split_input/' + pred_name)

            rv[pred_name] = make_gather_layer(gather_inds, pred_name)
            # rv[pred_name] should be batch_size*num_props_with_pred*1
            # rv[pred_name] = L.ConcatLayer(
            #     in_layers, axis=1, name='split_input/%s/concat' % pred_name)

        return rv

    def _split_extra(self, extra_data):
        """Sometimes we also have input data which goes straight to the
        network. We need to split this up into an unbound action->tensor
        dictionary just like the rest."""
        prob_meta = self._prob_meta
        out_dict = {}
        for unbound_act in prob_meta.domain.unbound_acts:
            ground_acts = prob_meta.schema_to_acts(unbound_act)
            sorted_acts = sorted(
                ground_acts, key=prob_meta.act_to_schema_subtensor_ind)
            if len(sorted_acts) == 0:
                # XXX: make this something scarier
                print("no actions for schema %s?" % unbound_act.schema_name)
            # these are the indices which we must read and concatenate
            tensor_inds = [
                # TODO: make this linear
                prob_meta.bound_acts_ordered.index(act) for act in sorted_acts
            ]

            # TODO: make commented stuff below work (e.g. by making sure all
            # ground actions are sorted by name or s.th)
            # start = min(tensor_inds)
            # stop = max(tensor_inds) + 1
            # approx_range = list(range(start, stop))
            # print('tensor_inds: ', tensor_inds)
            # print('approx_range: ', approx_range)
            # print('sorted_acts: ', sorted_acts)
            # print('bound_acts_ordered: ', prob_meta.bound_acts_ordered)
            # assert tensor_inds == approx_range, \
            #     "Order in which actions appear in input does not match " \
            #     "subtensor order."

            def python_closure_hatred(indices):
                """Runs a single tf.gather, for use within an OpLayer"""

                def inner(v):
                    return tf.gather(v, indices, axis=1)

                return inner

            def more_hate(tensor_inds):
                """Gives size of tensor returned by python_closure_hatred."""

                def inner(s):
                    return s[:1] + (len(tensor_inds), s[-1])

                return inner

            out_dict[unbound_act] = L.OpLayer(
                extra_data,
                python_closure_hatred(tensor_inds),
                more_hate(tensor_inds),
                name='split_extra/%s' % unbound_act.schema_name)

        return out_dict

    def _merge_finals(self, final_acts: Dict[UnboundAction, Any]) -> Any:
        prob_meta = self._prob_meta
        # we make a huge tensor of actions that we'll have to reorder
        sorted_final_acts = sorted(final_acts.items(), key=lambda t: t[0])
        # also get some metadata about which positions in tensor correspond to
        # which schemas
        unbound_to_super_ind = {
            t[0]: idx
            for idx, t in enumerate(sorted_final_acts)
        }
        # indiv_sizes[i] is the number of bound acts associated with the i-th
        # schema
        indiv_sizes = [
            len(prob_meta.schema_to_acts(ub)) for ub, _ in sorted_final_acts
        ]
        # cumul_sizes[i] is the sum of the number of ground actions associated
        # with each action schema *before* the i-th schema
        cumul_sizes = np.cumsum([0] + indiv_sizes)
        # this stores indices that we have to look up
        gather_list = []
        for ground_act in prob_meta.bound_acts_ordered:
            subact_ind = prob_meta.act_to_schema_subtensor_ind(ground_act)
            superact_ind = unbound_to_super_ind[ground_act.prototype]
            actual_ind = cumul_sizes[superact_ind] + subact_ind
            assert 0 <= actual_ind < prob_meta.num_acts, \
                "action index %d for %r out of range [0, %d)" \
                % (actual_ind, ground_act, prob_meta.num_acts)
            gather_list.append(actual_ind)

        # now let's actually build and reorder our huge tensor of action
        # selection probs
        cat_super_acts = L.ConcatLayer(
            [t[1] for t in sorted_final_acts], axis=1, name='merge_finals/cat')
        rv = L.OpLayer(
            incoming=cat_super_acts,
            # the [:, :, 0] drops the single dimension on the last axis
            op=lambda t: tf.gather(t[:, :, 0], np.array(gather_list), axis=1),
            shape_op=lambda s: s,
            name='merge_finals/reorder')

        return rv

    def _make_action_module(self,
                            prev_dict: Dict[str, Any],
                            unbound_act: UnboundAction,
                            output_size: int,
                            layer_num: int,
                            l_in: L.Layer,
                            nonlinearity: Any=None,
                            dropout: float=0.0,
                            norm_response=False,
                            extra_dict=None) -> Any:
        # TODO: can I do all of this index-futzing just once, instead of each
        # time I need to make an action module? Same applies to proposition
        # modules. Will make construction much faster (not that it's very
        # expensive at the moment...).
        name_pfx = 'act_mod_%s_%d' % (unbound_act.schema_name, layer_num)
        prob_meta = self._prob_meta
        dom_meta = self._weight_manager.dom_meta

        # sort input layers so we can pass them to Lasagne
        pred_to_tensor_idx, prev_inputs = self._sort_inputs(prev_dict)

        # this tells us how many channels our input will have to be
        index_spec = []
        dom_rel_preds = dom_meta.rel_pred_names(unbound_act)
        for act_pred_idx, arg_pred in enumerate(dom_rel_preds):
            pools = []
            for ground_act in prob_meta.schema_to_acts(unbound_act):
                # we're looking at the act_pred_idx-th relevant proposition
                bound_prop = prob_meta.rel_props(ground_act)[act_pred_idx]
                prop_idx = prob_meta.prop_to_pred_subtensor_ind(bound_prop)
                # we're only "pooling" over one element (the proposition
                # features)
                pools.append([prop_idx])

            # which tensor do we need to pick this out of?
            tensor_idx = pred_to_tensor_idx[arg_pred]
            index_spec.append((tensor_idx, pools))

        conv_input = PickPoolAndStack(
            prev_inputs, index_spec, name=name_pfx + '/cat')
        if layer_num == 0 and extra_dict is not None:
            # first action layer, so add in extra data
            act_data = extra_dict[unbound_act]
            conv_input = L.ConcatLayer(
                [conv_input, act_data], axis=2, name=name_pfx + '/extra_cat')
        W, b = self._weight_manager.act_weights[layer_num][unbound_act]
        if nonlinearity is None:
            nonlinearity = self.nonlinearity
        rv = L.Conv1DLayer(
            conv_input,
            output_size,
            filter_size=1,
            stride=1,
            pad='VALID',
            W=W,
            b=b,
            nonlinearity=nonlinearity,
            name=name_pfx + '/conv')
        if dropout > 0:
            rv = L.DropoutLayer(rv, p=dropout, name=name_pfx + '/drop')
        if norm_response and output_size > 1:
            rv = ResponseNormLayer(rv, name=name_pfx + '/norm')
        # BN won't work because it's a mess to apply in a net like this
        #     rv = L.BatchNormLayer(rv, center=True, scale=True)
        return rv

    def _make_prop_module(self,
                          prev_dict: Dict[UnboundAction, Any],
                          pred_name: str,
                          output_size: int,
                          layer_num: int,
                          l_in: Any,
                          norm_response: bool=False,
                          dropout: float=0.0):
        # TODO: comment about indexing futzing also applies here---should try
        # to do it just once.
        name_pfx = 'prop_mod_%s_%d' % (pred_name, layer_num)
        prob_meta = self._prob_meta
        dom_meta = self._weight_manager.dom_meta
        act_to_tensor_idx, prev_inputs = self._sort_inputs(prev_dict)
        index_spec = []  # type: List[Tuple[int, List[List[int]]]]
        pred_rela = dom_meta.rel_acts(pred_name)
        for rel_act_idx, rel_unbound_act in enumerate(pred_rela):
            pools = []
            for prop in prob_meta.pred_to_props(pred_name):
                # we're looking at the act_pred_idx-th relevant proposition
                ground_acts = [
                    ground_act
                    for unbound_act, ground_acts in prob_meta.rel_acts(prop)
                    for ground_act in ground_acts
                    if unbound_act == rel_unbound_act
                ]
                act_inds = [
                    prob_meta.act_to_schema_subtensor_ind(ground_act)
                    for ground_act in ground_acts
                ]
                pools.append(act_inds)

            tensor_idx = act_to_tensor_idx[rel_unbound_act]
            index_spec.append((tensor_idx, pools))

        conv_input = PickPoolAndStack(
            prev_inputs, index_spec, name=name_pfx + '/cat')
        W, b = self._weight_manager.prop_weights[layer_num][pred_name]
        rv = L.Conv1DLayer(
            conv_input,
            output_size,
            filter_size=1,
            stride=1,
            pad='VALID',
            W=W,
            b=b,
            nonlinearity=self.nonlinearity,
            name=name_pfx + '/conv')
        if dropout > 0:
            rv = L.DropoutLayer(rv, p=dropout, name=name_pfx + '/drop')
        if norm_response:
            rv = ResponseNormLayer(rv, name=name_pfx + '/norm')
        # this won't work because batch norm is a mess to apply here
        # rv = L.BatchNormLayer(rv, center=True, scale=True)
        return rv

    def _sort_inputs(self, prev_dict):
        # Sort order is kind of arbitrary. Main thing is that order implied by
        # pred_to_tensor_idx must exactly match that of prev_inputs, as
        # pred_to_tensor_idx is used to index into prev_inputs.
        input_items_sorted = sorted(prev_dict.items(), key=lambda p: p[0])
        pred_to_tensor_idx = {
            p[0]: idx
            for idx, p in enumerate(input_items_sorted)
        }
        prev_inputs = [tensor for _, tensor in input_items_sorted]
        return pred_to_tensor_idx, prev_inputs

    # input_layer and output_layer are used by CategoricalMLPPolicy
    @property
    def input_layer(self) -> Any:
        return self._l_in

    @property
    def output_layer(self) -> Any:
        return self._l_out
