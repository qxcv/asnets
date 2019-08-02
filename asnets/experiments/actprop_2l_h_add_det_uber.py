"""Two-layer config for solving harder deterministic problems."""

# use defaults from actprop_1l
from .actprop_1l_h_add import *  # noqa F401

NUM_LAYERS = 2
HIDDEN_SIZE = 12

# Hacky params
TIME_LIMIT_SECONDS = int(60 * 60 * 24)  # need lots of time due to plan costs
# TODO: need to change domain specifications so that they can have
# 'deterministic' flag that forces DET_EVAL to True
EVAL_ROUNDS = 1
ROUND_TURN_LIMIT = 500
DET_EVAL = True
