"""A two-layer configuration for the action/proposition network w/ Fast
Downward (FD) teacher & sparsity hacks."""

# use defaults from sparse 2-layer network
from .actprop_2l_h_add_sparse import *  # noqa F401

# deterministic eval, so only need one round
TEACHER_PLANNER = 'fd'
EVAL_ROUNDS = 1
