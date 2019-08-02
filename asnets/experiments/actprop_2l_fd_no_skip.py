"""A two-layer configuration for the action/proposition network w/ Fast
Downward (FD) teacher. No skip connections."""

# use defaults from actprop_2l_fd
from .actprop_2l_fd import *  # noqa F401

SKIP = False
