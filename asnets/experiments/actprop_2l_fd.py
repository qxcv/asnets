"""A two-layer configuration for the action/proposition network w/ Fast
Downward (FD) teacher."""

# use defaults from actprop_1l
from .actprop_1l import *  # noqa F401

NUM_LAYERS = 2
TEACHER_PLANNER = 'fd'
TRAINING_STRATEGY = 'ANY_GOOD_ACTION'
# deterministic eval, so only need one round
EVAL_ROUNDS = 1
DET_EVAL = True
# these probably help with GM, and might help with other things, too
USE_ACT_HISTORY_FEATURES = True
