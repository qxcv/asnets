"""A two-layer configuration for the action/proposition network w/ h-add
teacher. Uses no history, but does include LM-cut landmarks."""

from .actprop_2l_h_add import *  # noqa F401

USE_ACT_HISTORY_FEATURES = False
