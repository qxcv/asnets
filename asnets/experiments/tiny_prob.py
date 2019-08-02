"""For experiments on Triangle Tireworld."""

PDDL_DIR = '../problems/little-thiebaux/interesting/'
COMMON_PDDLS = ['triangle-tire.pddl']
TRAIN_PDDLS = ['triangle-tire-small.pddl']
TRAIN_NAMES = ['triangle-tire-1', 'triangle-tire-2']
TEST_RUNS = [
    (['triangle-tire-small.pddl'], 'triangle-tire-2'),
    (['triangle-tire-small.pddl'], 'triangle-tire-3'),
]  # yapf: disable
