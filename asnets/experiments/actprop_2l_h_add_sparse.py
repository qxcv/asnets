"""Variant of actprop_2l_h_add that is tuned for sparsity."""

# use defaults from actprop_2l_h_add
from .actprop_2l_h_add import *  # noqa F401

# removed dropout & made network narrower
NUM_LAYERS = 2
HIDDEN_SIZE = 8

# use a more aggressive learning rate schedule
SUPERVISED_LEARNING_RATE = 1e-2
LEARNING_RATE_STEPS = [(30, 1e-3), (60, 1e-4)]
# save very often
SAVE_EVERY_N_EPOCHS = 1
# don't early-stop
SUPERVISED_EARLY_STOP = 0

# aggressive L1 regularisation, but no L2 (this may have to be a bit different
# for each problem; might want to tune manually)
L1_REG = 1e-2
L2_REG = 0.0
# no dropout
DROPOUT = 0.0

# way more time, need to get to a better local min
TIME_LIMIT_SECONDS = int(60 * 60 * 8)
