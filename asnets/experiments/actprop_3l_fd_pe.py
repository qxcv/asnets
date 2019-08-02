"""A three-layer configuration for the action/proposition network w/ Fast
Downward (FD) teacher with probabilistic evaluation at test time (_pe)."""

from .actprop_3l_fd import *  # noqa F401

EVAL_ROUNDS = 10
DET_EVAL = False
