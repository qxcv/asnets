"""A two-layer configuration for the action/proposition network w/ FD
teacher, minus history features for network (but still with LM-cut)."""

# use defaults from actprop_2l_fd
from .actprop_2l_fd import *  # noqa F401

USE_ACT_HISTORY_FEATURES = False
