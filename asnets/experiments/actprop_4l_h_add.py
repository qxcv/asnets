"""A three-layer configuration for the action/proposition network w/ h-add
teacher."""

# use defaults from actprop_1l
from .actprop_1l_h_add import *  # noqa F401

NUM_LAYERS = 4
HIDDEN_SIZE = 12
