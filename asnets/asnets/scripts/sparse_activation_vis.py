#!/usr/bin/env python3
"""Visualise activations of a sparse policy network along trajectories produced
by the network."""

# stdlib imports
import argparse
import os
import os.path as path
import random

# third-party imports
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('cairo')  # required by graph_tool
import matplotlib.pyplot as plt  # noqa
import matplotlib.cm as cm  # noqa
import matplotlib.table as tbl  # noqa
import matplotlib.font_manager as fm  # noqa
# also holy shit graph_tool is a pain to install---you're on your own there!
import graph_tool.all as gt  # noqa

from asnets.interactive_network import NetworkInstantiator  # noqa
from asnets.prob_dom_meta import UnboundProp
from asnets.state_reprs import sample_next_state, get_init_cstate  # noqa
from asnets.scripts.weights2equations import snapshot_to_equations  # noqa

CMAP = cm.plasma


def roll_out_trajectory(policy, planner_exts, max_actions=20):
    """Runs the given policy from the initial state of a problem to the goal
    (or until the horizon given by `max_actions`)."""
    # see supervised.collect_paths for one example of how to do this
    cstates = [get_init_cstate(planner_exts)]
    enabled_acts_all = [cstates[-1].acts_enabled]
    act_layer_inputs_first = []
    act_layer_outputs = []
    prop_layer_outputs = []
    actions = []
    costs = []
    sess = tf.get_default_session()
    prob_meta = planner_exts.problem_meta
    while not cstates[-1].is_terminal and len(actions) < max_actions:
        this_state = cstates[-1]
        vec_state = [this_state.to_network_input()]
        next_act_dist_tensor, next_act_inputs, next_act_outputs, \
            next_prop_outputs = sess.run([
                policy.act_dist, policy.action_layer_input, policy.act_layers,
                policy.prop_layers
            ], feed_dict={policy.input_ph: vec_state})
        next_act_dist, = next_act_dist_tensor
        next_act_id = int(np.argmax(next_act_dist))
        act_layer_inputs_first.append(next_act_inputs)
        act_layer_outputs.append(next_act_outputs)
        prop_layer_outputs.append(next_prop_outputs)
        # we get BoundAction & add it to the trajectory
        # (use .unique_ident to convert it to something comprehensible)
        actions.append(prob_meta.bound_acts_ordered[next_act_id])
        next_state, cost = sample_next_state(cstates[-1], next_act_id,
                                             planner_exts)
        costs.append(cost)
        cstates.append(next_state)
        enabled_acts_all.append(next_state.acts_enabled)
    return cstates, actions, act_layer_inputs_first, act_layer_outputs, \
        prop_layer_outputs, costs, enabled_acts_all


def get_graph_ident(prop_or_act, equation):
    return '(%s)[%d]@L%d' % (prop_or_act.unique_ident, equation.out_la.index,
                             equation.out_la.layer)


class ReusableGraph(object):
    def __init__(self, *, old_graph=None):
        if old_graph:
            self.graph = old_graph.graph.copy()
            self.graph_pos = self.graph.vertex_properties.get("pos")
            self._nodes_by_id = {
                old_id: self.graph.vertex(int(old_node))
                for old_id, old_node in old_graph._nodes_by_id.items()
            }
        else:
            self.graph = gt.Graph()
            self._nodes_by_id = {}
            self.graph_pos = None
        self._get_vert_props("activation", "float")
        self._get_vert_props("global_layer", "int")
        self._get_vert_props("name", "string")
        self._get_vert_props("group", "string")
        self._get_vert_props("enabled", "bool")
        self._get_vert_props("chosen", "bool")
        self._get_vert_props("is_act", "bool")
        self._get_edge_props("is_rollout_act_edge", "bool")
        self._get_edge_props("eweight", "float")

    def copy(self):
        return self.__class__(old_graph=self)

    def _get_vert_props(self, name, dtype=None):
        if name not in self.graph.vertex_properties:
            assert dtype is not None, "need dtype to create props"
            self.graph.vertex_properties[name] \
                = self.graph.new_vertex_property(dtype)
        rv = self.graph.vertex_properties[name]
        setattr(self, 'graph_' + name, rv)
        return rv

    def _get_edge_props(self, name, dtype=None):
        if name not in self.graph.edge_properties:
            assert dtype is not None, "need dtype to create props"
            self.graph.edge_properties[name] \
                = self.graph.new_edge_property(dtype)
        rv = self.graph.edge_properties[name]
        setattr(self, 'graph_edge_' + name, rv)
        return rv

    def get_node(self, node_id):
        node = self._nodes_by_id.get(node_id)
        if node:
            return node
        assert not self.graph_pos, \
            "adding new node, even though this will invalidate layout! Bad!"
        new_node = self.graph.add_vertex()
        self._nodes_by_id[node_id] = new_node
        return new_node

    def set_node_properties(self,
                            node,
                            *,
                            activation,
                            global_layer,
                            name,
                            group,
                            is_act,
                            enabled=False,
                            chosen=False):
        self.graph_activation[node] = activation
        self.graph_global_layer[node] = global_layer
        self.graph_name[node] = name
        self.graph_group[node] = group
        self.graph_enabled[node] = enabled
        self.graph_chosen[node] = chosen
        self.graph_is_act[node] = is_act

    def add_edge(self, start_id, end_id):
        start = self._nodes_by_id[start_id]
        end = self._nodes_by_id[end_id]
        return self.graph.edge(start, end, add_missing=True)

    def set_edge_properties(self, edge, *, is_rollout_act_edge):
        self.graph_edge_is_rollout_act_edge[edge] = is_rollout_act_edge
        if is_rollout_act_edge:
            # give lower weight to rollout edges, since they reflect structure
            # of solution rather than structure of the problem
            self.graph_edge_eweight[edge] = 0.05
        else:
            self.graph_edge_eweight[edge] = 1.0

    def print_ALL_THE_EDGES(self):
        """Shows me what the edges are and what they mean. For debugging."""
        edge_tups = []
        ids_by_node = {
            node_num: node_str_id
            for node_str_id, node_num in self._nodes_by_id.items()
        }
        for edge in self.graph.edges():
            src = str(ids_by_node[int(edge.source())])
            dst = str(ids_by_node[int(edge.target())])
            edge_tups.append((src, dst))
        edge_tups = sorted(edge_tups)
        print('EDGES (%d):' % len(edge_tups))
        for src_str, dst_str in edge_tups:
            print('  %s -> %s' % (src_str, dst_str))
        rev_edges = sorted((dst, src) for src, dst in edge_tups)
        print('EDGES REVERSED (dest on left, source on right):')
        for dst_str, src_str in rev_edges:
            print('  %s <- %s' % (dst_str, src_str))

    def get_layout(self):
        if self.graph_pos is None:
            groups = self.graph_group
            sorted_uniq_groups = sorted(set(groups))
            group_names = {
                name: id
                for id, name in enumerate(sorted_uniq_groups)
            }
            group_ids = self.graph.new_vertex_property(
                "int",
                vals=[
                    group_names[g]
                    for g in self.graph.vertex_properties["group"]
                ])

            # I don't really know much about tuning this algorithm, but
            # empirically I've found that tightening up epsilon & cooling_step
            # gives much cleaner graphs.
            self.graph_pos = gt.sfdp_layout(
                self.graph,
                groups=group_ids,
                epsilon=1e-6,
                cooling_step=1 - 5e-3,
                eweight=self.graph_edge_eweight,
            )
            self.graph.vertex_properties['pos'] = self.graph_pos
        return self.graph_pos


def make_output_graph(sparse_equations,
                      act_inputs_first,
                      act_outputs,
                      prop_outputs,
                      problem_meta,
                      domain_meta,
                      enabled_acts,
                      chosen_act,
                      bound_act_seq,
                      draw_act_seq,
                      *,
                      original_rgraph=None,
                      verbose=False):
    n_act_layers = len(act_outputs)
    n_prop_layers = len(prop_outputs)
    assert len(sparse_equations) == n_act_layers + n_prop_layers

    enabled_acts = {act for act, is_enabled in enabled_acts if is_enabled}

    # this gets set inside the loop (I'm setting it to None up here to make
    # it clear that it gets used outside the loop)
    final_act_dict = None

    # output graph & graph properties
    og = ReusableGraph() if original_rgraph is None else original_rgraph.copy()

    act_inputs_first = {
        uact.schema_name: tensor[0]
        for uact, tensor in act_inputs_first.items()
    }

    for global_layer_num in range(n_act_layers + n_prop_layers):
        # layers 0, 2, 4, etc. are action; 1, 3, 5, etc. are proposition
        is_act = (global_layer_num % 2) == 0
        is_input_layer = global_layer_num == 0
        is_final_layer = global_layer_num == n_act_layers + n_prop_layers - 1
        equations = sparse_equations[global_layer_num]
        if is_act:
            modules_verbose = act_outputs[global_layer_num // 2]
            # modules are indexed by schema name
            modules = {
                ub_act.schema_name: out_tensor
                for ub_act, out_tensor in modules_verbose.items()
            }
            schema_name_to_ub_act = {
                ub_act.schema_name: ub_act
                for ub_act in modules_verbose.keys()
            }
        else:
            # modules are indexed by string-valued predicate name
            modules = prop_outputs[(global_layer_num - 1) // 2]

        # shear off leading (batch) dimension from each of the values in the
        # module dict
        for k in modules.keys():
            old_tensor = modules[k]
            shape = old_tensor.shape
            assert len(shape) == 3 and shape[0] == 1, \
                "tensor shape %s is weird" % (shape, )
            trimmed_tensor = old_tensor[0]
            modules[k] = trimmed_tensor

        # reset from last run through loop; on final iteration this will be
        # preserved for benefit of drawing code
        final_act_dict = {}

        for equation in equations:
            if verbose:
                print('Values for', equation)
            mod_name = equation.out_la.act_or_prop_id
            feat_maps = modules[mod_name]
            index = equation.out_la.index
            feat_map = feat_maps[:, index]
            used_inds = set()

            def add_edge(bound_prop=None, bound_act=None, this_key=None):
                """For adding edges to a newly-created node. This has been
                hoisted out into a function because the same code is needed for
                both the action module & proposition module branches below."""

                assert this_key is not None

                if bound_prop is not None:
                    destination_is_act = False
                    assert bound_act is None
                    rel_acts_by_schema_name_slot = {
                        (ub_act.schema_name, slot): bound_acts
                        for ub_act, slot, bound_acts in
                        problem_meta.rel_act_slots(bound_prop)
                    }
                else:
                    destination_is_act = True
                    assert bound_act is not None
                    arg_dict = dict(
                        zip(bound_act.prototype.param_names,
                            bound_act.arguments))
                for in_item in equation.in_la_list:
                    source_is_prop = in_item.role == 'pred'
                    sources_are_acts = in_item.role == 'act'
                    assert source_is_prop != sources_are_acts
                    if sources_are_acts:
                        # the input can be either an action or a proposition
                        if destination_is_act:
                            # skip conns, act->act
                            rel_acts = [bound_act]
                            last_layer = global_layer_num - 2
                        else:
                            # normal conns, act->prop
                            rel_acts = rel_acts_by_schema_name_slot[
                                (in_item.act_or_prop_id, in_item.slot)]
                            last_layer = global_layer_num - 1
                        for next_bound_act in rel_acts:
                            in_key = (next_bound_act, in_item.index,
                                      last_layer)
                            edge = og.add_edge(in_key, this_key)
                            og.set_edge_properties(
                                edge, is_rollout_act_edge=False)
                    else:
                        if destination_is_act:
                            ground_prop = in_item.backing_obj.bind(arg_dict)
                            last_layer = global_layer_num - 1
                        else:
                            # skip conn
                            ground_prop = bound_prop
                            last_layer = global_layer_num - 2
                        # can use in_item.index, in_item.layer
                        in_key = (ground_prop, in_item.index, last_layer)
                        edge = og.add_edge(in_key, this_key)
                        og.set_edge_properties(edge, is_rollout_act_edge=False)

            if is_act:
                unbound_act = schema_name_to_ub_act[mod_name]
                bound_acts = problem_meta.schema_to_acts(unbound_act)
                for bound_act in bound_acts:
                    subtens_ind = problem_meta \
                        .act_to_schema_subtensor_ind(bound_act)
                    feature_value = feat_map[subtens_ind]
                    used_inds.add(subtens_ind)

                    # print (optionally)
                    if verbose and not is_input_layer:
                        print('  %s: %.3f' % (bound_act.unique_ident,
                                              feature_value))
                    elif verbose and is_input_layer:
                        # Idea for doing this: in addition to saving
                        # activations at the output of each module, also try to
                        # save inputs for first layer. That way you can get at
                        # them with sess.run()
                        input_all_vals \
                            = act_inputs_first[mod_name][subtens_ind]
                        input_filt_vals = []
                        for in_act_name in equation.in_la_list:
                            input_val = input_all_vals[in_act_name.index]
                            input_filt_vals.append(input_val)
                        if not input_filt_vals:
                            input_str = 'no inputs'
                        else:
                            input_str = 'inputs [' + ', '.join(
                                ['%g' % in_val
                                 for in_val in input_filt_vals]) + ']'
                        print('  %s: %.3f, %s' % (bound_act.unique_ident,
                                                  feature_value, input_str))

                    # add to the graph
                    new_name = get_graph_ident(bound_act, equation)
                    this_key = (bound_act, index, global_layer_num)
                    final_act_dict[bound_act] = this_key
                    new_node = og.get_node(this_key)
                    og.set_node_properties(
                        new_node,
                        activation=feature_value,
                        global_layer=global_layer_num,
                        group=bound_act.unique_ident,
                        name=new_name,
                        is_act=True,
                        enabled=bound_act in enabled_acts and is_final_layer,
                        chosen=bound_act == chosen_act and is_final_layer)

                    # add edges from dependencies of this node to the current
                    # node
                    if global_layer_num > 0:
                        add_edge(bound_act=bound_act, this_key=this_key)

            else:
                bound_props = problem_meta.pred_to_props(mod_name)
                for bound_prop in bound_props:
                    subtens_ind = problem_meta \
                        .prop_to_pred_subtensor_ind(bound_prop)
                    feature_value = feat_map[subtens_ind]
                    used_inds.add(subtens_ind)

                    # print (optionally)
                    if verbose:
                        print('  %s: %.3f' % (bound_prop.unique_ident,
                                              feature_value))

                    # add to graph
                    this_key = (bound_prop, index, global_layer_num)
                    new_node = og.get_node(this_key)
                    new_name = get_graph_ident(bound_prop, equation)
                    og.set_node_properties(
                        new_node,
                        activation=feature_value,
                        global_layer=global_layer_num,
                        group=bound_prop.unique_ident,
                        name=new_name,
                        is_act=False,
                        enabled=False,
                        chosen=False)

                    add_edge(bound_prop=bound_prop, this_key=this_key)

            # sanity check
            assert used_inds == set(range(len(feat_map))), \
                "didn't use all the inds or used too many inds or sth"

    if draw_act_seq:
        # add in sequence of chosen actions
        for current_act, next_act in zip(bound_act_seq, bound_act_seq[1:]):
            current_id = final_act_dict[current_act]
            next_id = final_act_dict[next_act]
            new_edge = og.add_edge(current_id, next_id)
            og.set_edge_properties(new_edge, is_rollout_act_edge=True)

    return og


def add_colours(graphs, cmap):
    """Give each graph a set of `fill_colour` vertex attributes based on
    activations using the supplied cmap. Colours are normalised for consistency
    across all graphs within a given layer. For example, layer 1 in graph 1 and
    layer 1 in graph 2 use the same normalising transform, but layers 1 and 2
    in graphs 1 and 2 use different normalising transforms."""

    # first, join all activation maps into one long list
    prop_maps = (zip(graph.vertex_properties['activation'],
                     graph.vertex_properties['global_layer'])
                 for graph in graphs)
    all_activations_layers = sum(map(list, prop_maps), [])

    # now collect min/max by layer
    max_by_layer = {}
    min_by_layer = {}
    range_by_layer = {}
    for activation, layer in all_activations_layers:
        max_by_layer[layer] = max(activation,
                                  max_by_layer.get(layer, activation))
        min_by_layer[layer] = min(activation,
                                  min_by_layer.get(layer, activation))
        range_by_layer[layer] = max_by_layer[layer] - min_by_layer[layer]

    for graph in graphs:
        vertex_cols = graph.vertex_properties['fill_colour'] \
            = graph.new_vertex_property("vector<float>")
        activations = graph.vertex_properties['activation']
        layers = graph.vertex_properties['global_layer']
        for vertex in graph.vertex_index:
            layer = layers[vertex]
            activation = activations[vertex]
            min_act = min_by_layer[layer]
            act_range = range_by_layer[layer]
            if abs(act_range) < 1e-5:
                norm_act = 0.5
            else:
                norm_act = (activation - min_act) / act_range
            vertex_cols[vertex] = cmap(norm_act)


def main(args):
    np.random.seed(args.seed)
    gt.seed_rng(args.seed)
    tf.random.set_random_seed(args.seed)
    random.seed(args.seed)
    with tf.Session(graph=tf.Graph()) as sess:
        net_instantiator = NetworkInstantiator(
            args.snapshot,
            extra_ppddl=[args.domain_path],
            use_lm_cuts=args.use_lm_cut,
            use_history=args.use_history,
            heuristic_data_gen_name=args.heur_data_gen)
        net_container = net_instantiator.net_for_problem([args.problem_path])

        # now let's scoop up all the useful information we can!
        planner_exts = net_container.planner_exts
        policy = net_container.policy
        # weight_manager = net_instantiator.weight_manager
        domain_meta = net_container.single_problem.dom_meta
        problem_meta = net_container.single_problem.prob_meta
        sess.run(tf.global_variables_initializer())
        (cstates, actions, act_layer_inputs_all, act_layer_outputs_all,
         prop_layer_outputs_all, costs, enabled_acts_all) \
            = roll_out_trajectory(policy, planner_exts, args.horizon)

    # Get relevant eqns. This will be a list of weights for alternating
    # action/proposition layers.
    sparse_equations = snapshot_to_equations(args.snapshot)
    out_rgraph = None
    graph_steps = []

    for time_step in range(len(actions)):
        cstate = cstates[time_step]
        cost = costs[time_step]
        bound_act = actions[time_step]
        act_layer_outputs = act_layer_outputs_all[time_step]
        prop_layer_outputs = prop_layer_outputs_all[time_step]
        act_layer_inputs_first = act_layer_inputs_all[time_step]
        enabled_acts = enabled_acts_all[time_step]

        print('STEP', time_step)
        print('Action:', bound_act.unique_ident)
        print('State:', str(cstate))
        print('Cost:', cost)
        # print('Activations:')
        out_rgraph = make_output_graph(
            sparse_equations,
            act_layer_inputs_first,
            act_layer_outputs,
            prop_layer_outputs,
            problem_meta,
            domain_meta,
            enabled_acts,
            bound_act,
            actions,
            args.draw_act_seq,
            original_rgraph=out_rgraph,
            verbose=args.verbose)
        out_rgraph.get_layout()
        print('-' * 40)
        nospace_ba_ident = str(bound_act.unique_ident) \
            .replace(' ', '-') \
            .replace('/', '-')
        out_path = path.join(
            args.save_dir,
            'frame%02d-%s%s' % (time_step, nospace_ba_ident, args.ext))
        graph_steps.append({'graph': out_rgraph, 'out_path': out_path})

    add_colours([step['graph'].graph for step in graph_steps], cmap=CMAP)

    for state, action, graph_step in zip(cstates, actions, graph_steps):
        os.makedirs(args.save_dir, exist_ok=True)
        draw_rgraph(
            graph_step['graph'],
            graph_step['out_path'],
            args.size,
            state,
            action,
            actions,
            draw_state=args.draw_state,
            draw_action=args.draw_action,
            draw_cmap=args.draw_cmap,
            draw_vertex_labels=args.draw_labels)


def filter_colours(graph, vertex_colours, enabled):
    """Create new colour map in which things are transparent if they're not
    ancestors of something that's enabled"""
    # things that we should be able to reach
    targets = []
    for vind in graph.vertex_index:
        if enabled[vind]:
            targets.append(vind)
    # remove edges that we inserted to illustrate rollout
    graph = graph.copy()
    graph.set_fast_edge_removal()
    edges = list(graph.edges())
    for nz_ind in graph.edge_properties['is_rollout_act_edge'].a.nonzero()[0]:
        graph.remove_edge(edges[nz_ind])
    results = gt.shortest_distance(graph, target=targets)
    # initially we assume all vertices will be cleared; we exonerate vertices
    # if there is some sane distance from the vertex to an enabled action
    clear = set(graph.vertex_index)
    gt_int_max = 2**31 - 1
    for vind in graph.vertex_index:
        for target in targets:
            if results[vind][target] < gt_int_max:
                # we're good!
                clear.remove(vind)
                break
    new_colours = vertex_colours.copy()
    for vertex in clear:
        # This makes it transparent:
        # new_colours[vertex][3] = 0.0
        # This makes it solid white:
        new_colours[vertex][:3] = [1.0, 1.0, 1.0]
        # (Note to self: you can't do `new_colours[vertex][:3] = 0.5`. It only
        # works correctly when you assign a *list* (or slice) to the sliced
        # elements. This is probably a bug in graph_tool.)
    return new_colours


def draw_rgraph(out_rgraph, out_path, size, cstate, action, actions,
                draw_state, draw_action, draw_cmap, draw_vertex_labels):
    out_graph = out_rgraph.graph
    enabled_chosen = zip(out_graph.vertex_properties['enabled'],
                         out_graph.vertex_properties['chosen'])
    vertex_colour_map = {
        # (enabled, chosen): colour
        (False, False): 'gray',
        (True, False): 'coral',
        (True, True): 'greenyellow',
    }
    vertex_colours = out_graph.new_vertex_property(
        "string", vals=[vertex_colour_map[e_c] for e_c in enabled_chosen])
    vertex_pen_widths = out_graph.new_vertex_property(
        "float", vals=[2.0 if c == "gray" else 8.0 for c in vertex_colours])
    vertex_fill_colours = filter_colours(
        out_graph, out_graph.vertex_properties['fill_colour'],
        out_graph.vertex_properties['enabled'])
    is_act_edge_labels = out_graph.edge_properties["is_rollout_act_edge"]
    edge_default_colour = [0.179, 0.203, 0.210, 0.8]
    # salmon is [0.980, 0.502, 0.447, 1.0] (close to coral)
    # greenyellow ("traditional" chartreuse) is [0.678, 1.0, 0.184]
    edge_act_colour = [0.678, 1.0, 0.184]
    edge_colours = out_graph.new_edge_property("vector<float>", [
        edge_act_colour if is_act_edge else edge_default_colour
        for is_act_edge in is_act_edge_labels
    ])
    edge_widths = out_graph.new_edge_property(
        "float",
        [
            # make the edges illustrating rollout actions thicker
            8.0 if is_act_edge else 4.0 for is_act_edge in is_act_edge_labels
        ])
    edge_dash_style = out_graph.new_edge_property(
        "vector<float>",
        [
            # dash the edges for rollout actions
            [0.8, 0.3, 1] if is_act_edge else []
            for is_act_edge in is_act_edge_labels
        ])
    groups = out_graph.vertex_properties['group']
    layer_nums = out_graph.vertex_properties['global_layer']
    max_layer = max(layer_nums)
    vertex_sizes = out_graph.new_vertex_property(
        "float",
        vals=[
            40.0 * (layer_num / max_layer)**2 + 20 for layer_num in layer_nums
        ])
    groups_trimmed = groups.copy()
    represented_groups = set()
    sel_acts = {ba.unique_ident for ba in actions}
    for node_id in out_graph.vertex_index:
        old_value = groups_trimmed[node_id]
        if (old_value in represented_groups or layer_nums[node_id] < max_layer
                or old_value not in sel_acts or not draw_vertex_labels):
            groups_trimmed[node_id] = ''
        else:
            represented_groups.add(old_value)
    vertex_shapes = out_graph.new_vertex_property(
        "string",
        vals=[
            'square' if is_act else 'circle'
            for is_act in out_graph.vertex_properties['is_act']
        ])

    # now for fun drawing part!
    # (well, not that fun, since it's matplotlib)

    # for some reason the vector formats produce very differently-sized images
    # to the raster formats; I get around this with stupid DPI hacks (when what
    # I *should* be doing is figuring out why matplotlib ALWAYS DOES THE WORST
    # THING POSSIBLE)
    if os.path.splitext(out_path)[1] in {'.png', '.jpg', '.jpeg', '.gif'}:
        dpi = 150
    else:
        dpi = 80
    fig = plt.figure(
        figsize=(size[0] // dpi, size[1] // dpi),
        dpi=dpi,
        frameon=False,
    )
    # this weird hack (https://stackoverflow.com/a/9295367) means we don't have
    # to play around with bbox_inches="tight" etc.
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # axis('equal') is 100% necessary to not get warped squares
    plt.axis('equal')
    font_family = 'sans-serif'
    gt.graph_draw(
        out_graph,
        out_rgraph.get_layout(),
        # edge styling
        edge_pen_width=edge_widths,
        edge_color=edge_colours,
        edge_dash_style=edge_dash_style,
        # vertex styling
        vertex_shape=vertex_shapes,
        vertex_color=vertex_colours,
        vertex_pen_width=vertex_pen_widths,
        vertex_size=vertex_sizes,
        vertex_fill_color=vertex_fill_colours,
        # this just centers text (nothing fancy)
        vertex_text_position=0,
        vertex_text=groups_trimmed,
        vertex_text_offset=[-2.5, 0.0],
        vertex_font_family=font_family,
        vertex_font_size=24,
        # output_size=size,
        mplfig=fig,
    )

    # horrendous code to get current state & action displayed on plot
    if draw_state or draw_action:
        table_rows = []
        if draw_action:
            table_rows.append(['Action:', '(%s)' % action.unique_ident])
        if draw_state:
            table_rows.append([
                'State:', ', '.join([
                    '(%s)' % p.unique_ident for p, t in cstate.props_true if t
                ])
            ])
        if table_rows:
            text_props = {
                'size': 36,
                'family': font_family,
            }
            head_fontprops = fm.FontProperties(weight='bold', **text_props)
            cell_fontprops = fm.FontProperties(**text_props)
            sa_table = plt.table(
                cellText=table_rows,
                loc='upper left',
                cellLoc='left',
                colLoc='left')
            for (row, col), cell in sa_table.get_celld().items():
                cell.set_linewidth(0.0)
                if col == 0:
                    cell.set_text_props(fontproperties=head_fontprops)
                    cell.set_width(0.15)
                else:
                    cell.set_text_props(fontproperties=cell_fontprops)
                    cell.set_width(0.4)
            # sa_table.auto_set_column_width(1)
            ax.add_artist(sa_table)

    # some even more horrendous code to draw a bloody legend (MY EYES!)
    # (this ramp-drawing code adapted from
    # https://matplotlib.org/tutorials/colors/colormaps.html#miscellaneous)
    if draw_cmap:
        gradient = np.linspace(0, 1, 128)
        gradient = np.vstack((gradient, gradient))
        # draw in top left ("extent" is l,r,b,t)
        im_ax = plt.axes([0.05, 0.9, 0.2, 0.02])
        im_ax.imshow(
            gradient,
            cmap=CMAP,
            aspect='auto',
        )
        im_ax.axis('off')

    # finally, save the damn thing!
    plt.savefig(out_path)


parser = argparse.ArgumentParser()
parser.add_argument('snapshot', help='path to .pkl file containing weights')
parser.add_argument('domain_path', help='path to PDDL for domain')
parser.add_argument('problem_path', help='path to PDDL for problem')
parser.add_argument(
    '--save-dir',
    default=os.getcwd(),
    help='directory to save visualisations to (default: cwd)')
parser.add_argument(
    '--ext',
    choices=('.png', '.jpg', '.jpeg', '.pdf', '.gif', '.eps', '.ps', '.svg'),
    default='.png',
    help='extension for figures')
parser.add_argument(
    '--use-lm-cut',
    action='store_true',
    dest='use_lm_cut',
    default=False,
    help='use LM-cut (IDK why this info is not stored in weight mgr.)')
parser.add_argument(
    '--no-use-lm-cut',
    action='store_false',
    dest='use_lm_cut',
    help='do not use LM-cut')
parser.add_argument(
    '--use-history',
    action='store_true',
    dest='use_history',
    default=False,
    help='use history features')
parser.add_argument(
    '--no-use-history',
    action='store_false',
    dest='use_history',
    help='do not use history features')
parser.add_argument(
    '--heur-data-gen',
    default=None,
    type=str,
    help='name of heuristic data generator for network input')
parser.add_argument(
    '--horizon',
    default=30,
    type=int,
    help='maximum # of actions in a trajectory')
parser.add_argument(
    '--verbose',
    default=False,
    action='store_true',
    help='print more output (e.g precise activation values)')
parser.add_argument(
    '--draw-state',
    default=False,
    action='store_true',
    help='draw state on plot')
parser.add_argument(
    '--draw-cmap',
    default=False,
    action='store_true',
    help='draw chosen colour map on plot')
parser.add_argument(
    '--no-draw-action',
    dest='draw_action',
    default=True,
    action='store_false',
    help="don't draw current action on plot")
parser.add_argument(
    '--no-draw-labels',
    dest='draw_labels',
    default=True,
    action='store_false',
    help="don't draw action labels on each vertex")
parser.add_argument(
    '--draw-act-seq',
    default=False,
    action='store_true',
    help='draw a series of arrows through successive selected actions')
parser.add_argument(
    '--size',
    nargs=2,
    type=int,
    default=(1200, 1200),
    help='size of output plot, in pixels')
parser.add_argument(
    '--seed', type=int, default=42, help='seed for Numpy, GraphTool etc.')


def _main():
    main(parser.parse_args())


if __name__ == '__main__':
    _main()
