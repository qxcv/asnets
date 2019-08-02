"""A three-layer configuration for the action/proposition network w/ FD
teacher, minus LM-cut features for network."""

# use defaults from actprop_3l_fd
from .actprop_3l_fd import *  # noqa F401

USE_LMCUT_FEATURES = False
