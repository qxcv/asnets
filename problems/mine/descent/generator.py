#!/usr/bin/env python3
"""Generator for mountaineer problems."""

import argparse
import os

import numpy as np


def visualise_graph(matrix):
    """Visualise graph and print most important statistics. Requires NetworkX,
    GraphViz, PyGraphViz, and Matplotlib."""
    import networkx as nx
    import matplotlib.pyplot as plt

    n = matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(n))
    for i in range(n):
        for j in range(n):
            if matrix[i, j]:
                G.add_edge(i, j)

    print()
    print('Vertices: ', G.number_of_nodes())
    print('Edges: ', G.number_of_edges())
    print('Shortest path:', nx.shortest_path_length(G, 0, n - 1))
    print('Longest path:', nx.algorithms.dag.dag_longest_path_length(G))
    print()
    assert G.number_of_nodes() == n
    assert nx.algorithms.dag.is_directed_acyclic_graph(G)

    # old plotting code:
    # nx.draw(G, with_labels=True)
    # plt.draw()
    # plt.show()

    # new DAG code: https://stackoverflow.com/a/11484144
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.show()


def generate_adjacency_matrix(rand_state, vertices):
    """Generate DAG adjacency matrix where everything is reachable from source
    node and can reach sink node. `vertices` is the number of node, `max_jump`
    is how far forward any one edge can connect vertices."""
    assert vertices >= 2, "need at least 2 vertices"
    max_jump = max(2, min(vertices - 2, 3))
    matrix = np.zeros((vertices, vertices), dtype=bool)
    for i in range(vertices - 1):
        # choose an outdegree
        max_outdeg = min(vertices - i - 1, max_jump)
        min_outdeg = 1
        # p(d) prop to 1/d
        out_dist = 1 / np.arange(1, max_outdeg + 1)
        out_dist = out_dist / np.sum(out_dist)
        outdeg = rand_state.choice(
            np.arange(min_outdeg, max_outdeg + 1), p=out_dist)

        # figure out which of the `outdeg` nodes we're going to jump to
        choice_up = min(i + 1 + max_jump, vertices)
        choices = np.arange(i + 1, choice_up)
        out_conns = rand_state.permutation(choices)[:outdeg]
        matrix[[i] * outdeg, out_conns] = True

        # if nothing connects to us, connect to immediate parent
        if i > 0 and np.sum(matrix[:, i]) == 0:
            matrix[i - 1, i] = True

    # some sanity checks: outdegree >= 1, indegree >= 1
    indegrees = np.sum(matrix, axis=0)
    assert np.all(indegrees[1:] >= 1), indegrees
    outdegrees = np.sum(matrix, axis=1)
    assert np.all(outdegrees[:-1] >= 1), outdegrees

    return matrix


def format_problem(mat, name):
    n = mat.shape[0]
    assert n > 0
    rv = []

    # prelude
    rv.append('(define (problem %s)' % name)
    rv.append('  (:domain descent)')

    # all locations
    digits = len(str(n - 1))
    loc_names = ['l' + str(i).zfill(digits) for i in range(n)]
    locs = ' '.join(loc_names)
    rv.append('  (:objects %s - location)' % locs)

    # start state
    rv.append('  (:init (at %s) (alive) (have-rope)' % loc_names[0])
    for start in range(n):
        for end in range(start + 1, n):
            if mat[start, end]:
                rv.append('    (descent %s %s)' %
                      (loc_names[start], loc_names[end]))
    rv.append('  )')

    # goal
    rv.append('  (:goal (and (at %s) (alive)))' % loc_names[-1])

    # close it off
    rv.append(')')
    rv.append('')

    return '\n'.join(rv)


parser = argparse.ArgumentParser(
    description='generate random mountaineer problems')
parser.add_argument('n', type=int, help='vertices in graph')
parser.add_argument('seed', type=int, help='seed for RNG')
parser.add_argument(
    '--vis', action='store_true', default=False, help='show image of graph')
parser.add_argument(
    '--dest-dir',
    default=None,
    help='directory to store in (otherwise just print)')


def main():
    args = parser.parse_args()
    state = np.random.RandomState((args.seed << 5) + args.n)
    mat = generate_adjacency_matrix(state, args.n)
    if args.vis:
        print(mat)
        visualise_graph(mat)
    name = 'descent-n%i-s%i' % (args.n, args.seed)
    prob_str = format_problem(mat, name)
    if args.dest_dir is not None:
        os.makedirs(args.dest_dir, exist_ok=True)
        dest_path = os.path.join(args.dest_dir, name + '.pddl')
        print('Writing to %s' % dest_path)
        with open(dest_path, 'w') as fp:
            fp.write(prob_str)
    else:
        print(prob_str)


if __name__ == '__main__':
    main()
