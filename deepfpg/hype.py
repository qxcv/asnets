#!/usr/bin/env python3
"""Hyperparameter tuning with hyperopt."""

from argparse import ArgumentParser
import json
from os import path
import re
from subprocess import check_output, CalledProcessError
from sys import stderr
import pickle

import hyperopt as hopt
import numpy as np

_UP_RE = re.compile(r'^Unique prefix: (.*)$', flags=re.MULTILINE)


def get_results(log_dir, output):
    """Process output of fpg.py and fetch corresponding results from log
    directory."""
    matches = _UP_RE.findall(output)
    if not matches:
        print('Could not find prefix in results')
    pfx = matches[0]
    results_path = path.join(log_dir, pfx + '-results.json')
    with open(results_path, 'r') as fp:
        return json.load(fp)


class PlannerInstance:
    """Class to store DFPG parameters which don't change between runs."""

    def __init__(self, args):
        self.log_dir = path.join(args.work_dir, 'logs')
        self.eval_rounds = args.eval_rounds
        self.turns = args.turns
        self.time_limit = args.time_limit
        self.pddl_paths = args.pddl_paths
        self.problem = args.problem
        self.score_weight = args.score_weight
        self.model_type = args.model_type

    def objective(self, args):
        """This is intended to be given to hyperopt's fmin."""
        num_layers, hidden_size, learning_rate, batch_size = args
        num_layers = int(num_layers)
        hidden_size = int(hidden_size)
        batch_size = int(batch_size)

        print('')
        print('Attempting run with:')
        print('num_layers=%d' % num_layers)
        print('hidden_size=%d' % hidden_size)
        print('learning_rate=%f' % learning_rate)
        print('batch_size=%d' % batch_size)
        print('')

        if self.model_type == 'actprop':
            model_args = [
                "-m", "actprop", "-O",
                "num_layers=%d,hidden_size=%d" % (num_layers, hidden_size)
            ]
        else:
            model_args = [
                "-m", "simple", "-O",
                "num_layers=%d,hidden_size=%d" % (num_layers, hidden_size)
            ]
        cmdline = [
            "./fpg.py", "-p", self.problem, "-t", str(self.time_limit), "-l",
            self.log_dir, '-L', str(self.turns), '-R', str(self.eval_rounds),
            "-f", "-o", "vpg", "-A",
            "learning_rate=%f,batch_size=%d" % (learning_rate, batch_size),
            *model_args, *self.pddl_paths
        ]
        try:
            proc_stdout = check_output(cmdline, universal_newlines=True)
        except CalledProcessError as e:
            print(
                'Error running fpy.py on %s: %s\n\nFailed command: %s' %
                (self.problem, str(e), cmdline),
                file=stderr)
            # BUG: I think that when the first run fails, hyperopt crashes on
            # hyperopt/base.py:585 when it tries to get the minimum loss over
            # all runs. Presumably this does not matter for later runs.
            return {'loss': 1e100, 'status': hopt.STATUS_FAIL}
        results = get_results(self.log_dir, proc_stdout)
        opt_time = results['elapsed_opt_time']
        succ_rate = results['successes'] / float(results['trials'])
        fail_rate = 1 - succ_rate
        hp_loss = fail_rate * self.score_weight + opt_time
        print('Success rate from this round: %f' % succ_rate)
        return {'loss': hp_loss, 'status': hopt.STATUS_OK}


parser = ArgumentParser()
parser.add_argument(
    '--resume-from', default=None, help='path to file to resume from')
parser.add_argument(
    '--work-dir', help='working directory', default='hyperopt/')
parser.add_argument(
    '--eval-rounds', help='rounds for evaluation', type=int, default=100)
# It's probably possible to figure out what this should be automatically: find
# maximum achievable success rate and use twice the time for that run (for
# instance) as the weighting.
parser.add_argument(
    '--score-weight',
    help='score weight for goodness computation',
    type=float,
    # a policy which never gets anything done gets a 1h penalty; penalty scales
    # linearly down from there
    default=3600)
parser.add_argument('--turns', help='turns per round', type=int, default=100)
parser.add_argument(
    '--time-limit',
    help='number of seconds to wait for convergence',
    type=float,
    default=900.0)
parser.add_argument(
    '--model-type',
    choices=['actprop', 'simple'],
    default='actprop',
    help='type of network to use (see fpg.py help)')
parser.add_argument(
    'problem', nargs='?', help='name of problem to optimise')
parser.add_argument('pddl_paths', nargs='*', help='paths to .pddls')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.resume_from is not None:
        with open(args.resume_from, 'rb') as fp:
            loaded = pickle.load(fp)
        pi = loaded['pi']
        space = loaded['space']
        trials = loaded['trials']
        epoch = loaded['epoch'] + 1
        print('Loaded from resume point at %s (epoch %d)' %
              (args.resume_from, epoch))
    else:
        assert len(args.pddl_paths) > 0, "Need some paths"
        assert args.problem is not None, "Need a given problem"
        pi = PlannerInstance(args)
        # TODO: make these configurable
        space = [
            hopt.hp.quniform('num_layers', 1, 2, 1),
            hopt.hp.quniform('hidden_size', 8, 64, 1),
            hopt.hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            hopt.hp.qloguniform('batch_size', np.log(10), np.log(4000), 1),
        ]
        trials = hopt.Trials()
        epoch = 0

    print('Starting hyperparameter search from epoch %d' % epoch)
    while True:
        extra = 1
        best = hopt.fmin(
            fn=pi.objective,
            space=space,
            algo=hopt.tpe.suggest,
            # do <extra> evals
            max_evals=len(trials) + extra,
            trials=trials,
            verbose=True)
        print('Epoch {} completed. Best params: {}'.format(epoch, best))
        to_dump = {
            'epoch': epoch,
            'best': best,
            'space': space,
            'pi': pi,
            'trials': trials
        }
        pkl_path = path.join(args.work_dir, 'checkpoint.pkl')
        with open(pkl_path, 'wb') as fp:
            pickle.dump(to_dump, fp)
        print('Dumped to %s' % pkl_path)
        epoch += 1
