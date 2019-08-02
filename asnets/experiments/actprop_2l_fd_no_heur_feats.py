"""A two-layer configuration for the action/proposition network w/ FD
teacher, minus LM-cut features for network."""

# use defaults from actprop_2l_fd
from .actprop_2l_fd import *  # noqa F401

USE_LMCUT_FEATURES = False
USE_ACT_HISTORY_FEATURES = False
