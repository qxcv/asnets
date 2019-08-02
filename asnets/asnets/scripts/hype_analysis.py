"""Analysis script for hyperparameter tuning experiment."""

import glob
import json
import os
import pprint
import sys

import click
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis, \
    unnest_checkpoints
from ray.tune.error import TuneError
from ray.tune.trial_runner import _TuneFunctionDecoder


class PatchedExperimentAnalysis(ExperimentAnalysis):
    """Patched version of ExperimentAnalysis that fixes two bugs:

    1. json.load() does not correctly un-pickle things that have been stored
       with CloudPickle by the trial runner. To fix this, I just use the
       correct JSON decoder.
    2. if experiment_path is relative, then loading will break for reasons I
       don't fully understand. I think it's doing path.join(experiment_path,
       experiment_path_read_from_json) somewhere, but I'm not sure where that
       could be happening. I fix this by just making the path absolute in all
       cases."""

    def __init__(self, experiment_path, trials=None):
        # no super().__init__() b/c ExperimentAnalysis constructor is broken
        # (we're overwriting it)
        experiment_path = os.path.abspath(os.path.expanduser(experiment_path))
        if not os.path.isdir(experiment_path):
            raise TuneError(
                "{} is not a valid directory.".format(experiment_path))
        experiment_state_paths = glob.glob(
            os.path.join(experiment_path, "experiment_state*.json"))
        if not experiment_state_paths:
            raise TuneError(
                "No experiment state found in {}!".format(experiment_path))
        if len(experiment_state_paths) > 1:
            print(
                "WARNING: %d tune state files found, taking only last one" %
                len(experiment_state_paths),
                file=sys.stderr)
        experiment_filename = max(
            list(experiment_state_paths))  # if more than one, pick latest
        with open(experiment_filename) as f:
            self._experiment_state = json.load(f, cls=_TuneFunctionDecoder)

        if "checkpoints" not in self._experiment_state:
            raise TuneError("Experiment state invalid; no checkpoints found.")
        self._checkpoints = self._experiment_state["checkpoints"]
        self._scrubbed_checkpoints = unnest_checkpoints(self._checkpoints)
        self.trials = trials
        self._dataframe = None


@click.command()
@click.option(
    '--directory',
    default='./experiment-results/tune/perform_trial/',
    help="path to Tune's perform_trial directory")
@click.option(
    '--out',
    default='tune-results.csv',
    help='path to store results in (default: tune-results.csv)')
def main(directory, out):
    ex = PatchedExperimentAnalysis(directory)

    best_info = ex.get_best_info(metric='coverage', mode='max', flatten=False)
    print(best_info.keys())
    coverage = best_info['last_result']['coverage']
    config = best_info['config']
    print("Best config with coverage of %g:" % coverage)
    pprint.pprint(config)

    print("Saving full dataframe to '%s'" % out)
    df = ex.dataframe()
    df.to_csv(out)


if __name__ == '__main__':
    main()
