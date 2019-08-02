"""Towers of Hanoi. I just prepared this as a test of ASNets' plan
representation abilities, so it's hacky (trains on the test set, doesn't
evaluate on large enough problems to choke the baselines, etc.)."""

PDDL_DIR = '../problems/mine/hanoi/'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    'problems/hanoi-3.pddl',
    'problems/hanoi-4.pddl',
    'problems/hanoi-5.pddl',
    'problems/hanoi-6.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['problems/hanoi-3.pddl'], None),
    (['problems/hanoi-4.pddl'], None),
    (['problems/hanoi-5.pddl'], None),
    (['problems/hanoi-6.pddl'], None),
    (['problems/hanoi-7.pddl'], None),
    (['problems/hanoi-8.pddl'], None),
]  # yapf: disable
