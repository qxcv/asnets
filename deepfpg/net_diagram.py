#!/usr/bin/env python3
"""Draw a rough diagram of an action-proposition network."""

import argparse
from colorsys import hsv_to_rgb, rgb_to_hsv
from hashlib import sha512

import graphviz as gv
from matplotlib import cm
import numpy as np

from models import get_domain_meta, get_problem_meta
from mdpsim_utils import parse_problem_args


def safe_merge(d1, d2):
    """Merge two dictionaries together, first checking for duplicates."""
    assert d1.keys().isdisjoint(d2)
    rv = dict(d1)
    rv.update(d2)
    return rv


def strident(thing):
    hash = sha512(thing.encode('utf8'))
    return hash.hexdigest()


def mix_rgbs(rgb_strs):
    rgbs = list(map(str2rgb, rgb_strs))
    mean = np.mean(rgbs, axis=0)
    rv = rgb2str(mean)
    return rv


def rgb2str(rgb):
    assert len(rgb) == 3
    rgb_arr = np.array(rgb) * 255
    rgb_arr[rgb_arr < 0] = 0
    rgb_arr[rgb_arr > 255] = 255
    rgb_ints = map(int, rgb_arr)
    rgb_str = '#{:02x}{:02x}{:02x}'.format(*rgb_ints)
    return rgb_str


def str2rgb(str):
    assert len(str) == 7 and str[0] == '#'
    rh, bh, gh = str[1:3], str[3:5], str[5:]
    return tuple(int(v, 16) / 255.0 for v in [rh, bh, gh])


class VisualPropNetwork():
    def __init__(self, problem_meta, domain_meta, num_hidden):
        self._nhidden = num_hidden
        self._prob_meta = problem_meta
        self._dom_meta = domain_meta
        self.dot = gv.Digraph()
        self._node_set = set()
        self._edge_set = set()
        self._merge_ctr = 1
        self.node_colours = {}

        def make_hue_val_map(items, cmap):
            rv = {}
            sort_idxs_items = enumerate(sorted(items))
            for idx, item in sort_idxs_items:
                hsv = rgb_to_hsv(*cmap(idx % cmap.N)[:-1])
                rv[item] = (hsv[0], hsv[1])
            return rv

        self.act_schema_hue_val = make_hue_val_map(
            self._prob_meta.domain.unbound_acts, cm.Set1)
        self.pred_hue_val = make_hue_val_map(
            self._prob_meta.domain.pred_names, cm.Set2)

        def make_rgb_str_map(major_to_minor, major_map):
            rv = {}
            for major, minors in major_to_minor.items():
                hue, val = major_map[major]
                bot = 0.3
                top = 0.7
                sat_arr = np.linspace(bot, top, len(minors))
                if len(sat_arr) == 1:
                    # if we don't do this then we just get bot :(
                    sat_arr = np.mean([bot, top], keepdims=True)
                for sat, minor in zip(sat_arr, sorted(minors)):
                    assert minor not in rv, "dupe '%s'" % minor
                    rgb = hsv_to_rgb(hue, sat, val)
                    rgb_str = rgb2str(rgb)
                    rv[minor] = rgb_str
            return rv

        schema_to_act_map = {}
        self.act_colour = make_rgb_str_map(schema_to_act_map,
                                           self.act_schema_hue_val)
        pred_to_prop_map = {}
        self.prop_colour = make_rgb_str_map(pred_to_prop_map,
                                            self.pred_hue_val)

        self.make_graph()

    def _add_node(self, name, *args, **kwargs):
        # dupe avoiding wrapper around G.add_node
        if name in self._node_set:
            raise ValueError("Duplicate node '%s'!" % name)
        if 'color' in kwargs:
            self.node_colours[name] = kwargs['color']
        rv = self.dot.node(strident(name), *args, **kwargs)
        self._node_set.add(name)
        return rv

    def prop_node(self, name, layer_num):
        if layer_num is None:
            full_name = '%s:input' % name
        else:
            full_name = '%s:%d:prop' % (name, layer_num)
        colour = self.prop_colour[name]
        self._add_node(
            full_name,
            label=name,
            shape='box',
            color=colour,
            style='setlinewidth(4)')
        return full_name

    def act_node(self, name, layer_num):
        full_name = '%s:%d:act' % (name, layer_num)
        colour = self.act_colour[name]
        self._add_node(
            full_name,
            label=name,
            shape='box',
            color=colour,
            style='setlinewidth(4)')
        return full_name

    def merge(self, inputs, type='concat'):
        # I'm only using mean/concat in the original
        type_labels = {
            'mean': '+',
            'concat': '[]',
        }
        assert type in type_labels.keys(), \
            'are you sure you wanted a "%s" merge?' % type
        assert len(inputs) >= 1, "no inputs to merge(?!)"
        # no point doubling up
        if len(inputs) == 1:
            # no need to merge at all
            input, = inputs
            return input
        full_name = '%s:%d:merge' % (type, self._merge_ctr)
        input_dict = {}
        colours = []
        for input in inputs:
            input_dict.setdefault(input, 0)
            input_dict[input] += 1
            if input in self.node_colours:
                colours.append(self.node_colours[input])
        kwargs = dict(label=type_labels[type], shape='circle')
        if len(colours) > 0:
            # need to mix
            kwargs['color'] = mix_rgbs(colours)
        self._add_node(full_name, **kwargs)
        self._merge_ctr += 1
        for input, count in input_dict.items():
            self.add_edge(input, full_name, dupes=count)
        return full_name

    def add_edge(self, node1, node2, dupes=1, colour=None):
        tup = (node1, node2)
        if tup in self._edge_set:
            raise ValueError("duplicate edge {}".format(tup))
        else:
            kwargs = {}
            kwargs['style'] = 'setlinewidth(%f)' % dupes
            if node1 in self.node_colours:
                kwargs['color'] = self.node_colours[node1]
            self.dot.edge(strident(node1), strident(node2), **kwargs)
            self._edge_set.add(tup)

    def cluster(self, nodes, name):
        with self.dot.subgraph(name='cluster_%s' % strident(name)) as c:
            for n in set(nodes):
                c.node(strident(n))
            c.attr(label=name)
            c.attr(color='black')

    def make_graph(self):
        dom_meta = self._dom_meta
        prob_meta = self._prob_meta
        # this is just input data
        prop_dict = {
            prop_name: self.prop_node(prop_name, None)
            for prop_name in prob_meta.input_props_ordered
        }
        self.cluster(prop_dict.values(), 'Input propositions')

        # hidden layers
        for hid_idx in range(self._nhidden):
            act_dict = {}
            for schema_name in dom_meta.rel_preds.keys():
                new_acts = self.action_module(prop_dict, schema_name, hid_idx)
                act_dict = safe_merge(act_dict, new_acts)
            self.cluster(act_dict.values(), 'Action layer %d' % hid_idx)

            prop_dict = {}
            for pred_name in dom_meta.rel_acts.keys():
                new_props = self.prop_module(act_dict, pred_name, hid_idx)
                prop_dict = safe_merge(prop_dict, new_props)
            self.cluster(prop_dict.values(), 'Proposition layer %d' % hid_idx)

        # final (action) layer
        act_dict = {}
        for schema_name in dom_meta.rel_preds.keys():
            new_acts = self.action_module(prop_dict, schema_name,
                                          self._nhidden)
            act_dict = safe_merge(act_dict, new_acts)
        self.cluster(act_dict.values(), 'Final action layer')

        # extra graph attributes
        self.dot.attr(ranksep='8')

    def action_module(self, prev_dict, schema_name, layer_num):
        prob_meta = self._prob_meta
        rv = {}

        for ground_act in prob_meta.schema_to_acts[schema_name]:
            merge_inputs = {}
            for prop_name in prob_meta.rel_props[ground_act]:
                current = merge_inputs.get(prev_dict[prop_name], 0)
                merge_inputs[prev_dict[prop_name]] = current + 1

            rv[ground_act] = self.act_node(ground_act, layer_num)
            for input, dupes in merge_inputs.items():
                self.add_edge(input, rv[ground_act], dupes=dupes)

        return rv

    def prop_module(self, prev_dict, pred_name, layer_num):
        rv = {}
        prob_meta = self._prob_meta
        dom_meta = self._dom_meta
        for prop in prob_meta.pred_to_props[pred_name]:
            concat_inputs = {}
            for rel_schema in dom_meta.rel_acts[pred_name]:
                mean_inputs = [
                    prev_dict[ground_act]
                    for ground_act in prob_meta.rel_acts[prop]
                    if prob_meta.act_to_schema[ground_act] == rel_schema
                ]
                if not mean_inputs:
                    # Sometimes an action is enabled in some instances but not
                    # in others, so we only end up feeding it into some nodes
                    # (e.g. change-tire in TTW). In this case we just continue.
                    continue
                merge_label = self.merge(mean_inputs, type='mean')
                concat_inputs.setdefault(merge_label, 0)
                if merge_label in concat_inputs:
                    concat_inputs[merge_label] += 1
                else:
                    concat_inputs[merge_label] = 0
            rv[prop] = self.prop_node(prop, layer_num)
            for input, dupes in concat_inputs.items():
                self.add_edge(input, rv[prop], dupes=dupes)

        return rv

    def save(self, dest):
        # save an image of the graph
        self.dot.render(dest, cleanup=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--problem',
    default=None,
    help='name of problem to make a network for')
parser.add_argument(
    '-n',
    '--num-hidden',
    type=int,
    default=1,
    help='number of layers in network')
parser.add_argument(
    '-o', '--output', default='out', help='output filename prefix')
parser.add_argument(
    'pddls', nargs='+', help='paths to PDDL domain/problem definitions')

if __name__ == '__main__':
    args = parser.parse_args()
    mdpsim = __import__('mdpsim')
    problem = parse_problem_args(mdpsim, args.pddls, args.problem)
    print('Selected problem %s' % problem)
    dom_meta = get_domain_meta(problem.domain)
    prob_meta = get_problem_meta(problem, dom_meta)
    vpn = VisualPropNetwork(prob_meta, dom_meta, args.num_hidden)
    print('Saving to %s' % args.output)
    vpn.save(args.output)
