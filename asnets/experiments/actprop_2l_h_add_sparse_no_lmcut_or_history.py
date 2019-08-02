"""A sparse two-layer configuration for the action/proposition network w/ h-add
teacher and no heuristic inputs at all."""

# use defaults from actprop_2l_h_add_sparse
from .actprop_2l_h_add_sparse import *  # noqa F401

USE_LMCUT_FEATURES = False
USE_ACT_HISTORY_FEATURES = False
