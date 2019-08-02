#!/usr/bin/env python3
"""Visualise a nice grid of correlations between propositions and output action
logits."""

from argparse import ArgumentParser
from json import loads

import matplotlib.pyplot as plt
import numpy as np


def trial_paths_to_matrix(tp_path):
    """Open a trial-paths.txt file and turn it into matching observation and
    action dist matrices."""
    all_inits = []
    all_dists = []
    with open(tp_path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            # split out the X/Y in "X -> Y -> Z" (and ignore the Z at the end)
            json_parts = line.split(" -> ")[:-1]
            parts = [loads(part) for part in json_parts]
            inits_dists = [(p["init"], p["dist"]) for p in parts]
            inits, dists = zip(*inits_dists)
            all_inits.extend(inits)
            all_dists.extend(dists)
    return np.array(all_inits), np.array(all_dists)


def corr_matrix(inits, dists):
    """Calculates correlation coefficients between init variables and dist
    variables."""
    # mean-centre both matrices
    init_cen = inits - inits.mean(axis=0)[None]
    dists_cen = dists - dists.mean(axis=0)[None]
    # compute covariance between relevant variable
    n = init_cen.shape[0]
    assert n > 1, \
        "can't calculate sample covariance with one or fewer samples"
    init_std = init_cen.std(axis=0)[None]
    init_std[init_std <= 1e-7] = 1
    dists_std = dists_cen.std(axis=0)[None]
    dists_std[dists_std <= 1e-7] = 1
    norm_init = init_cen * (init_std > 1e-7) / init_std
    norm_dists = dists_cen * (dists_std > 1e-7) / dists_std
    ucorr = np.dot(norm_init.T, norm_dists)
    corr = ucorr / (n - 1)
    assert corr.shape == (init_cen.shape[1], dists_cen.shape[1])
    # done!
    return corr


def vis_corr_matrix(corr):
    """Produce a nice grid visualisation of correlation matrix. Based on
    sklearn's confusion matrix example."""
    plt.imshow(corr, interpolation='nearest')  # , cmap=plt.cm.Blues)
    # plt.xticks(np.arange(corr.shape[1]))
    # plt.yticks(np.arange(corr.shape[0]))
    plt.xticks([])
    plt.yticks([])


parser = ArgumentParser()
parser.add_argument(
    'trial_paths',
    nargs='+',
    help='path to trial_paths.txt; last is assumed to be most recent')


def main():
    args = parser.parse_args()
    cmats = []
    for tp_path in args.trial_paths:
        inits, dists = trial_paths_to_matrix(tp_path)
        cmat = corr_matrix(inits, dists)
        cmats.append(cmat)
    # get sort order for actions
    sort_inds_actions = np.argsort(np.abs(cmats[-1]).max(axis=0))
    cmats = [cmat[:, sort_inds_actions] for cmat in cmats]
    # get sort order for state variables
    last_cmat = np.abs(cmats[-1])
    max_corr = np.argmax(last_cmat, axis=1)
    sort_inds = np.argsort(max_corr)
    last_cmat = last_cmat[sort_inds]
    nz_inds = np.nonzero((last_cmat != 0).any(axis=1))
    # now rejig
    cmats = [cmat[sort_inds][nz_inds] for cmat in cmats]
    # now display!
    n = len(cmats)
    for i, cmat in enumerate(cmats, start=1):
        plt.subplot(n, 1, i)
        vis_corr_matrix(cmat.T)
        if i != n:
            plt.title('Attention matrix before training')
        else:
            plt.title('Attention matrix after training')
        plt.xlabel('Input state variables')
        plt.ylabel('Output actions')
    plt.show()


if __name__ == '__main__':
    main()
