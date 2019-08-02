"""For experiments on our CosaNostra Pizza domain. This file is meant for use
in the sparse learning experiments."""

PDDL_DIR = '../problems/mine/cosanostra/'
COMMON_PDDLS = ['cosanostra.pddl']
TRAIN_PDDLS = [
    # remove two smallest problems to avoid need for special-case logic
    # 'problems/cosanostra-n1.pddl',
    # 'problems/cosanostra-n2.pddl',
    'problems/cosanostra-n3.pddl',
    'problems/cosanostra-n4.pddl',
    'problems/cosanostra-n5.pddl'
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    (['problems/cosanostra-n6.pddl'], None),
    (['problems/cosanostra-n7.pddl'], None),
    (['problems/cosanostra-n8.pddl'], None),
    (['problems/cosanostra-n9.pddl'], None),
    (['problems/cosanostra-n10.pddl'], None),
    (['problems/cosanostra-n11.pddl'], None),
    (['problems/cosanostra-n12.pddl'], None),
    (['problems/cosanostra-n13.pddl'], None),
    (['problems/cosanostra-n14.pddl'], None),
    (['problems/cosanostra-n15.pddl'], None),
    (['problems/cosanostra-n20.pddl'], None),
    (['problems/cosanostra-n25.pddl'], None),
    (['problems/cosanostra-n30.pddl'], None),
    (['problems/cosanostra-n35.pddl'], None),
    # also remove big eval files; we don't really care about thorough eval for
    # the sparsity experiments
    # (['problems/cosanostra-n40.pddl'], None),
    # (['problems/cosanostra-n45.pddl'], None),
    # (['problems/cosanostra-n50.pddl'], None),
    # (['problems/cosanostra-n55.pddl'], None),
    # (['problems/cosanostra-n60.pddl'], None),
    # (['problems/cosanostra-n65.pddl'], None),
    # (['problems/cosanostra-n70.pddl'], None),
    # (['problems/cosanostra-n75.pddl'], None),
    # (['problems/cosanostra-n80.pddl'], None),
    # (['problems/cosanostra-n85.pddl'], None),
    # (['problems/cosanostra-n90.pddl'], None),
    # (['problems/cosanostra-n95.pddl'], None),
]  # yapf: disable
