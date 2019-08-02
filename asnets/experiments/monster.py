"""For experiments on our monster domain."""

PDDL_DIR = '../problems/mine/monster/'
COMMON_PDDLS = ['monster.pddl']
TRAIN_PDDLS = [
    'problems/monster-n1.pddl',
    'problems/monster-n2.pddl',
    'problems/monster-n3.pddl',
    'problems/monster-n4.pddl',
    'problems/monster-n5.pddl',
    'problems/monster-n6.pddl',
    'problems/monster-n7.pddl',
    'problems/monster-n8.pddl',
    'problems/monster-n9.pddl',
    'problems/monster-n10.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    # we train and test on the same problem set, since we're only interested in
    # whether optimal policy is in the net's policy space
    (['problems/monster-n1.pddl'], None),
    (['problems/monster-n2.pddl'], None),
    (['problems/monster-n3.pddl'], None),
    (['problems/monster-n4.pddl'], None),
    (['problems/monster-n5.pddl'], None),
    (['problems/monster-n6.pddl'], None),
    (['problems/monster-n7.pddl'], None),
    (['problems/monster-n8.pddl'], None),
    (['problems/monster-n9.pddl'], None),
    (['problems/monster-n10.pddl'], None),
]  # yapf: disable
