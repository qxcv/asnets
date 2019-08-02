"""A two-layer configuration for the action/proposition network w/ Fast
Downward (FD) teacher. Specialised to help it solve det BW, etc."""

# use defaults from actprop_1l
from .actprop_1l import *  # noqa F401

NUM_LAYERS = 2
TEACHER_PLANNER = 'fd'
TRAINING_STRATEGY = 'ANY_GOOD_ACTION'
SUPERVISED_LEARNING_RATE = 1e-3
LEARNING_RATE_STEPS = []
HIDDEN_SIZE = 20
TARGET_ROLLOUTS_PER_EPOCH = 150
# this should help with exploration
DROPOUT = 0.3
# deterministic eval, so only need one round
EVAL_ROUNDS = 1
DET_EVAL = True
# these probably help with GM, and might help with other things, too
USE_ACT_HISTORY_FEATURES = True
# give it a bit more time do deal with planning
TIME_LIMIT_SECONDS = int(60 * 60 * 6)
# don't early-stop
SUPERVISED_EARLY_STOP = 9999
