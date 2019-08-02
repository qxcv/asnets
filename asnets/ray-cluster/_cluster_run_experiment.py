#!/usr/bin/env python3
"""This Python script is meant to be submitted to a cluster with 'ray submit'.
Its only purpose is to ensure we're in the correct directory to run
experiments. See cluster_asnet_experiments.sh for usage."""

import os

from asnets.scripts.run_experiment import main


if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/l4p/asnets/'))
    main()
