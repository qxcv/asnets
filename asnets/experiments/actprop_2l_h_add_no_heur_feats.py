"""A two-layer configuration for the action/proposition network w/ h-add
teacher."""

# use defaults from actprop_2l_h_add
from .actprop_2l_h_add import *  # noqa F401

USE_LMCUT_FEATURES = False
USE_ACT_HISTORY_FEATURES = False
