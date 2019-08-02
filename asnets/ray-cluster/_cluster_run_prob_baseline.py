#!/usr/bin/env python3
import os

from asnets.scripts.run_prob_baselines import main


if __name__ == '__main__':
    os.chdir(os.path.expanduser('~/l4p/asnets/'))
    main()
