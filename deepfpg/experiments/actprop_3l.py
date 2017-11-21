"""A two-layer configuration for the action/proposition network."""

# use defaults from actprop_1l
from .actprop_1l import *  # noqa F401

TRAIN_MODEL_FLAGS = 'num_layers=3,hidden_size=16,dropout=0.25'
TEST_MODEL_FLAGS = 'num_layers=3,hidden_size=16'
