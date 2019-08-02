"""A two-layer configuration for the action/proposition network w/
domain-specific (DS) teacher."""

# use defaults from actprop_1l
from .actprop_1l import *  # noqa F401

NUM_LAYERS = 2
TEACHER_PLANNER = 'domain-specific'
# deterministic eval, so only need one round
EVAL_ROUNDS = 1
DET_EVAL = True
