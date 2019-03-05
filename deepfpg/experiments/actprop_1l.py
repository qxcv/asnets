"""One-layer configuration for action-proposition network"""

# train supervised or RL? (only supervised supported at the moment)
SUPERVISED = True

# learning rate
SUPERVISED_LEARNING_RATE = 0.0001
# batch size
SUPERVISED_BATCH_SIZE = 128
# heuristic for supervised teacher
SUPERVISED_TEACHER_HEURISTIC = 'lm-cut'

# model flags
MODEL_TYPE = 'actprop'
TRAIN_MODEL_FLAGS = 'num_layers=1,hidden_size=16,dropout=0.25'
TEST_MODEL_FLAGS = 'num_layers=1,hidden_size=16'
USE_LMCUT_FEATURES = True

# train + eval settings
TIME_LIMIT_SECONDS = int(60 * 60 * 1)   # XXX turned down to 1h for monster
EVAL_ROUNDS = 30
ROUND_TURN_LIMIT = 300
DET_EVAL = True
