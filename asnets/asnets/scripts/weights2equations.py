#!/usr/bin/env python3
"""Convert a set of weights (probably sparse weights) into a set of equations
describing what those weights do."""

import argparse

import joblib
import numpy as np
import tensorflow as tf

from asnets.prob_dom_meta import UnboundAction


def load_weights(weight_path):
    # use throw-away session with a temporary graph so that we don't clutter
    # main graph
    with tf.Session(graph=tf.Graph()) as sess:
        wm = joblib.load(weight_path)
        sess.run(tf.global_variables_initializer())
        # wm_state is a dict containing the following keys:
        # dom_meta (domain metadata), hidden_sizes (sizes of hidden layers),
        # extra_dim (size of extra heuristic inputs for each action),
        # prop_weights_np (prop layer weights as numpy arrays), and
        # act_weights_np (action layer weights as numpy arrays).
        wm_state = wm.__getstate__()
    return wm_state


def modify_leaves(f, ds):
    """Walks a data structure composed of nested lists/tuples/dicts, and
    creates a new (equivalent) structure in which the leaves of the data
    structure (i.e the non-list/tuple/dict data structures in the tree) have
    been mapped through f."""
    if isinstance(ds, list):
        return [modify_leaves(f, sds) for sds in ds]
    if isinstance(ds, tuple):
        return tuple(modify_leaves(f, sds) for sds in ds)
    if isinstance(ds, dict):
        return {k: modify_leaves(f, sds) for k, sds in ds.items()}
    return f(ds)


def simplify_weight_matrix(weight_mat, sparse_thresh=1e-2):
    """Simplify weight matrix so that it is either of size [out_dim, in_dim] or
    just [out_dim], then zero out the tiniest weights."""

    # first make shape right
    if weight_mat.ndim == 2:
        # default shape is (in_dim, out_dim), per _make_conv_weights in
        # models.py
        new_mat = weight_mat.T.copy()
    else:
        assert weight_mat.ndim == 1, "shape %s is weird" % (weight_mat.shape, )
        new_mat = weight_mat.copy()

    # zero out all tiny weights (they generally don't contribute anything
    # anyway)
    zero_inds = np.abs(new_mat) < sparse_thresh
    new_mat[zero_inds] = 0.0

    return new_mat


def simplify_np_weights(weights):
    return modify_leaves(simplify_weight_matrix, weights)


class LiftedActivationName:
    # Represents a single activation in an action/proposition module, but not
    # bound to a *specific* action or proposition. Useful for tracking
    # information related to inputs/outputs of a lifted action or proposition
    # modules.
    def __init__(self,
                 name,
                 act_or_prop_id,
                 layer,
                 index,
                 backing_obj=None,
                 role=None,
                 slot=None):
        self.name = name
        self.act_or_prop_id = act_or_prop_id
        self.layer = layer
        self.index = index
        self.backing_obj = backing_obj
        self.role = role
        self.slot = slot
        NoneType = type(None)
        assert isinstance(self.act_or_prop_id, (str, NoneType)), \
            self.act_or_prop_id
        assert isinstance(self.layer, (int, NoneType)), self.layer
        assert isinstance(self.index, (int, NoneType)), self.index

    def ident(self):
        # FIXME: should handle different groupings when pooling (I think? This
        # ought to be important for the graph thing).
        return (self.act_or_prop_id, self.layer, self.index)

    def __str__(self):
        return self.name


class Equation:
    def __init__(self, out_la, in_la_list, coeffs, bias, activation='elu'):
        assert isinstance(out_la, LiftedActivationName)
        assert isinstance(in_la_list, list) and all(
            isinstance(la, LiftedActivationName) for la in in_la_list)
        assert len(in_la_list) == len(coeffs)
        self.out_la = out_la
        self.in_la_list = in_la_list
        self.coeffs = [float(coeff) for coeff in coeffs]
        assert len(self.coeffs) == len(self.in_la_list)
        self.bias = float(bias)
        self.activation = activation

    def prune_small_coeffs(self, thresh=1e-3):
        new_la_list = []
        new_coeff_list = []
        for coeff, la in zip(self.coeffs, self.in_la_list):
            if np.abs(coeff) >= thresh:
                new_coeff_list.append(coeff)
                new_la_list.append(la)
        return Equation(
            self.out_la,
            new_la_list,
            new_coeff_list,
            self.bias,
            activation=self.activation)

    def replace_with_const(self, term, constant):
        that_ident = term.ident()
        new_bias = self.bias
        new_in_la_list = []
        new_coeff_list = []
        seen = False
        for coeff, my_la in zip(self.coeffs, self.in_la_list):
            if my_la.ident() == that_ident:
                # actually double-appearances are okay in general because they
                # might refer to different bound terms (e.g at(?l1) vs at(?l2)
                # in drive(?l1,?l2)).
                # assert not seen, \
                #     'double appearance of %s in %s' % (term, self)
                new_bias += coeff * constant
                seen = True
            else:
                new_in_la_list.append(my_la)
                new_coeff_list.append(coeff)
        assert seen, 'no appearance of %s in %s' % (term, self)
        return Equation(
            self.out_la,
            new_in_la_list,
            new_coeff_list,
            new_bias,
            activation=self.activation)

    def appears_on_rhs(self, la):
        # does this equation contain la on RHS?
        that_ident = la.ident()
        for my_la in self.in_la_list:
            if my_la.ident() == that_ident:
                return True
        return False

    @property
    def is_constant(self):
        # sometimes the RHS of an equation is just a constant b/c everything
        # other term has been pruned
        return len(self.in_la_list) == 0

    def __str__(self):
        parts = []
        num_format = '%.3f'
        for coeff, in_la in zip(self.coeffs, self.in_la_list):
            parts.append(num_format % coeff + '*' + in_la.name)
        if np.abs(self.bias) > 0 or not parts:
            parts.append(num_format % self.bias)
        out_name = self.out_la.name
        rhs = ' + '.join(parts)
        lhs = out_name + ' = '
        if self.activation:
            out_str = '%s%s(%s)' % (lhs, self.activation, rhs)
        else:
            out_str = lhs + rhs
        return out_str


def get_input_output_names(mod_type, mod_label, layer_num, hidden_sizes,
                           extra_dim, dom_meta):
    """Get a list of pretty input & output names for the given module."""
    num_act_layers = len(hidden_sizes) + 1
    num_prop_layers = len(hidden_sizes)

    # check numbering convention
    assert layer_num >= 0, "layers start at 0"
    if mod_type == 'act':
        assert layer_num <= num_act_layers
        assert isinstance(mod_label, UnboundAction)
    elif mod_type == 'prop':
        assert layer_num <= num_prop_layers
        # we just get a string name for prop layers (which is fine—we don't
        # need access to arguments anyway, since bindings are ignored at prop
        # layers)
        assert isinstance(mod_label, str)
    else:
        raise ValueError("Unknown mod type '%s'" % (mod_type, ))

    input_names = []
    if mod_type == 'act':
        base_out_name = '%s(%s)' % (mod_label.schema_name, ','.join(
            mod_label.param_names))
        # this is a list of predicates w/ bindings
        input_predicates = mod_label.rel_props
        # useful things:
        # UnboundAction.{schema_name,param_names,rel_props}
        # UnboundProp.{pred_name,params}
        input_pred_names = [
            '%s(%s)' % (input_pred.pred_name, ','.join(input_pred.params))
            for input_pred in input_predicates
        ]
        if layer_num == 0:
            # need special handling for proposition layers' outputs
            # order of inputs to this layer is given by unbound_act.props
            new_names = []
            for pred_name in input_pred_names:
                new_names.append('is_true(%s)' % pred_name)
                new_names.append('is_goal(%s)' % pred_name)
            new_names.extend('heuristic_data(%s)[%d]' % (
                base_out_name,
                extra_idx,
            ) for extra_idx in range(extra_dim))
            # we don't need any metadata about the inputs
            input_names.extend(
                LiftedActivationName(name, None, None, index=index)
                for index, name in enumerate(new_names))
        else:
            for input_pred, pred_name in zip(input_predicates,
                                             input_pred_names):
                last_hid_size = hidden_sizes[layer_num - 1][1]
                input_names.extend([
                    LiftedActivationName(
                        '%s[%d,%d]' % (pred_name, layer_num - 1, in_idx),
                        input_pred.pred_name,
                        layer_num - 1,
                        in_idx,
                        backing_obj=input_pred,
                        role='pred') for in_idx in range(last_hid_size)
                ])
            # skip connections
            prev_act_hid_size = hidden_sizes[layer_num - 1][0]
            input_names.extend([
                LiftedActivationName(
                    '%s[%d,%d]' % (base_out_name, layer_num - 1, in_idx),
                    input_pred.pred_name,
                    layer_num - 1,
                    in_idx,
                    backing_obj=mod_label,
                    role='act') for in_idx in range(prev_act_hid_size)
            ])
    else:
        base_out_name = '%s(…)' % (mod_label, )
        rel_ub_act_slots = dom_meta.rel_act_slots(mod_label)
        hid_act_size = hidden_sizes[layer_num][0]
        for uba, slot in rel_ub_act_slots:
            input_names.extend(
                LiftedActivationName(
                    'pool(%s(…)@%s)[%d,%d]' %
                    (uba.schema_name, slot, layer_num, in_idx),
                    uba.schema_name,
                    layer_num,
                    in_idx,
                    role='act', slot=slot) for in_idx in range(hid_act_size))
        if layer_num > 0:
            # skip connections
            prev_prop_hid_size = hidden_sizes[layer_num - 1][1]
            input_names.extend(
                LiftedActivationName(
                    '%s[%d,%d]' % (mod_label, layer_num - 1, in_idx),
                    mod_label,
                    layer_num - 1,
                    in_idx,
                    backing_obj=None,
                    role='pred') for in_idx in range(prev_prop_hid_size))

    if mod_type == 'act' and layer_num + 1 == num_act_layers:
        # need special handling for final action layers' outputs
        num_outputs = 1
    else:
        # hidden size is (act size, prop size)
        num_outputs = hidden_sizes[layer_num][mod_type != "act"]

    out_id_name = mod_label.schema_name if mod_type == 'act' else mod_label

    # final activation is linear
    output_names = [
        LiftedActivationName(
            '%s[%d,%d]' % (base_out_name, layer_num, out_idx),
            out_id_name,
            layer_num,
            out_idx,
            backing_obj=mod_label) for out_idx in range(num_outputs)
    ]

    return input_names, output_names


def build_single_equations(input_names,
                           output_names,
                           weight_matrix,
                           bias_vec,
                           last_layer=False):
    assert weight_matrix.shape == (len(output_names), len(input_names)), \
        "weight matrix shape %s but %d inputs and %d outputs" % (
            weight_matrix.shape, len(input_names), len(output_names))
    equations = []
    for output_num, output_name in enumerate(output_names):
        weight_row = weight_matrix[output_num]
        assert len(weight_row) == len(input_names)
        bias = bias_vec[output_num]
        activation = None if last_layer else 'elu'
        equations.append(
            Equation(
                output_name,
                input_names,
                weight_row,
                bias,
                activation=activation))
    return equations


def numpy_elu(x):
    x = float(x)
    if x >= 0:
        return x
    return np.exp(x) - 1


def build_equations_raw(act_weights, prop_weights, dom_meta, extra_dim,
                        hidden_sizes):
    layered_equations = []

    # step 1: get all the equations
    for layer_num in range(len(hidden_sizes) + 1):
        # act layers
        act_equations = []
        for act_label in dom_meta.unbound_acts:
            act_input_names, act_output_names = get_input_output_names(
                'act', act_label, layer_num, hidden_sizes, extra_dim, dom_meta)
            act_weight_mat, act_bias = act_weights[layer_num][act_label]
            act_equations.extend(
                build_single_equations(
                    act_input_names,
                    act_output_names,
                    act_weight_mat,
                    act_bias,
                    last_layer=layer_num == len(hidden_sizes)))
        layered_equations.append(act_equations)

        if layer_num < len(hidden_sizes):
            # prop layer too
            prop_equations = []
            for pred_label in dom_meta.pred_names:
                pred_input_names, pred_output_names = get_input_output_names(
                    'prop', pred_label, layer_num, hidden_sizes, extra_dim,
                    dom_meta)
                pred_weight_mat, pred_bias \
                    = prop_weights[layer_num][pred_label]
                prop_equations.extend(
                    build_single_equations(pred_input_names, pred_output_names,
                                           pred_weight_mat, pred_bias))
            layered_equations.append(prop_equations)

    # step 2: prune all the zeroed coefficients
    layered_equations = [[eqn.prune_small_coeffs() for eqn in eqn_list]
                         for eqn_list in layered_equations]

    # step 3: lift constant values from each prop layer into the next action
    # layer, instead of having them appear as separate terms. Note that we
    # can't lift from action layer into proposition layer because there's no
    # guarantee that there is at least one action related to the corresponding
    # proposition! (also more generally that would not work if we were doing
    # average pooling, or any pooling mechanism where pooling over zero inputs
    # yields a different output to pooling over n inputs, even if the inputs
    # are constant)
    # FIXME: also consider lifting biases from one action or proposition layer
    # to the next wherever skip connections exist (assuming that becomes an
    # actual problem)
    for prop_layer, next_action_layer in zip(layered_equations[1::2],
                                             layered_equations[2::2]):
        for eqn in prop_layer:
            if not eqn.is_constant:
                continue
            # replace every occurrence of eqn in next_layer
            const_value = numpy_elu(eqn.bias)
            lhs = eqn.out_la
            for idx, next_eqn in enumerate(next_action_layer):
                if not next_eqn.appears_on_rhs(lhs):
                    continue
                next_action_layer[idx] = next_eqn.replace_with_const(
                    lhs, const_value)

    # step 4: remove equation for x in layer l if there's no y in layer l+1
    # that depends on x, or x'' in layer l+2 that depends on x (via a skip
    # connection)
    new_layered_equations = []
    for layer_idx in range(len(layered_equations) - 1):
        this_layer = layered_equations[layer_idx]
        next_layer = layered_equations[layer_idx + 1]
        look_layers = list(next_layer)
        if layer_idx < len(layered_equations) - 2:
            # also look for outbound skip conns
            look_layers.extend(layered_equations[layer_idx + 2])
        new_this_layer = []
        for this_eqn in this_layer:
            # does this equation appear in any of the next layer's equations?
            if any(
                    next_eqn.appears_on_rhs(this_eqn.out_la)
                    for next_eqn in next_layer):
                new_this_layer.append(this_eqn)
        new_layered_equations.append(new_this_layer)
    new_layered_equations.append(layered_equations[-1])
    layered_equations = new_layered_equations

    return layered_equations


def equations_to_strings(layered_equations):
    equation_lines = []
    for this_layer in layered_equations:
        equation_lines.extend(map(str, this_layer))
        equation_lines.append('')

    return equation_lines


def snapshot_to_equations(pkl_path):
    # load up the pickle
    weight_data = load_weights(pkl_path)

    # extra data
    dom_meta = weight_data['dom_meta']
    extra_dim = weight_data['extra_dim']
    hidden_sizes = weight_data['hidden_sizes']

    # simplify weights (make them sane shape, induce some sparsity)
    prop_weights = weight_data['prop_weights_np']
    act_weights = weight_data['act_weights_np']
    prop_weights = simplify_np_weights(prop_weights)
    act_weights = simplify_np_weights(act_weights)

    # now build equations & print them out
    equations_raw = build_equations_raw(act_weights, prop_weights, dom_meta,
                                        extra_dim, hidden_sizes)
    return equations_raw


def main(args):
    equations = snapshot_to_equations(args.weights)
    equation_strs = equations_to_strings(equations)

    print('Equations for this network:')
    print('\n'.join(map(str, equation_strs)))


parser = argparse.ArgumentParser()
parser.add_argument('weights', help='path to .pkl file containing weights')

if __name__ == '__main__':
    main(parser.parse_args())
