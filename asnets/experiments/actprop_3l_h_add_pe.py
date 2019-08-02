"""A three-layer configuration for the action/proposition network w/ h-add
teacher & probabilistic evaluation (hence "_pe" at end of name)."""

# use defaults from actprop_3l
from .actprop_3l_h_add import *  # noqa F401

# stochastic evaluation!
DET_EVAL = False
EVAL_ROUNDS = 30
