"""For experiments on Triangle Tireworld."""

PDDL_DIR = '../problems/little-thiebaux/interesting/'
COMMON_PDDLS = ['triangle-tire.pddl']
TRAIN_PDDLS = ['triangle-tire-small.pddl']
TRAIN_NAMES = ['triangle-tire-1', 'triangle-tire-2', 'triangle-tire-3']
TEST_RUNS = [
    (['triangle-tire-small.pddl'], 'triangle-tire-4'),
    (['triangle-tire-small.pddl'], 'triangle-tire-5'),
    # 6 onwards have their own files (because they get big fast)
    (['ttw-extra/triangle-tire-6.pddl'], 'triangle-tire-6'),
    (['ttw-extra/triangle-tire-7.pddl'], 'triangle-tire-7'),
    (['ttw-extra/triangle-tire-8.pddl'], 'triangle-tire-8'),
    (['ttw-extra/triangle-tire-9.pddl'], 'triangle-tire-9'),
    (['ttw-extra/triangle-tire-10.pddl'], 'triangle-tire-10'),
    (['ttw-extra/triangle-tire-11.pddl'], 'triangle-tire-11'),
    (['ttw-extra/triangle-tire-12.pddl'], 'triangle-tire-12'),
    (['ttw-extra/triangle-tire-13.pddl'], 'triangle-tire-13'),
    (['ttw-extra/triangle-tire-14.pddl'], 'triangle-tire-14'),
    (['ttw-extra/triangle-tire-15.pddl'], 'triangle-tire-15'),
    (['ttw-extra/triangle-tire-16.pddl'], 'triangle-tire-16'),
    (['ttw-extra/triangle-tire-17.pddl'], 'triangle-tire-17'),
    (['ttw-extra/triangle-tire-18.pddl'], 'triangle-tire-18'),
    (['ttw-extra/triangle-tire-19.pddl'], 'triangle-tire-19'),
    (['ttw-extra/triangle-tire-20.pddl'], 'triangle-tire-20'),
    # Last few take ages to evaluate with ASNets (they're absolutely huge
    # problems, so ASNets are slow and baselines have no chance at all)
    # (['ttw-extra/triangle-tire-25.pddl'], 'triangle-tire-25'),
    # (['ttw-extra/triangle-tire-30.pddl'], 'triangle-tire-30'),
    # (['ttw-extra/triangle-tire-35.pddl'], 'triangle-tire-35'),
]  # yapf: disable
