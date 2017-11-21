"""For experiments Felipe's modified SAR domain."""

# XXX: only works on planbox-huge@NeCTAR
PDDL_DIR = '/mnt/modified_sar_split/'
COMMON_PDDLS = ['domain.ppddl']
TRAIN_PDDLS = [
    # random selection 7 training problems on 4x4 grids
    'SAR_n4_v1_min-d4_pr0.25_P10.ppddl',
    'SAR_n4_v2_min-d4_pr0.50_P25.ppddl',
    'SAR_n4_v2_min-d4_pr0.75_P17.ppddl',
    'SAR_n4_v3_min-d4_pr0.25_P21.ppddl',
    'SAR_n4_v3_min-d4_pr0.75_P08.ppddl',
    'SAR_n4_v5_min-d4_pr0.25_P14.ppddl',
    'SAR_n4_v6_min-d4_pr0.25_P06.ppddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    # Random selection of 20 test problems
    (['SAR_n5_v4_min-d4_pr0.10_P05.ppddl'], None),
    (['SAR_n5_v8_min-d4_pr0.10_P09.ppddl'], None),
    (['SAR_n5_v9_min-d4_pr0.10_P18.ppddl'], None),
    (['SAR_n6_v10_min-d4_pr0.50_P01.ppddl'], None),
    (['SAR_n8_v8_min-d4_pr0.75_P10.ppddl'], None),
    (['SAR_n8_v9_min-d4_pr0.25_P04.ppddl'], None),
    (['SAR_n8_v10_min-d4_pr0.75_P22.ppddl'], None),
    (['SAR_n9_v2_min-d4_pr0.10_P24.ppddl'], None),
    (['SAR_n10_v1_min-d4_pr0.10_P13.ppddl'], None),
    (['SAR_n10_v6_min-d4_pr0.25_P27.ppddl'], None),
    (['SAR_n11_v7_min-d4_pr0.10_P16.ppddl'], None),
    (['SAR_n13_v5_min-d4_pr0.75_P21.ppddl'], None),
    (['SAR_n13_v7_min-d4_pr0.50_P17.ppddl'], None),
    (['SAR_n13_v10_min-d4_pr0.50_P20.ppddl'], None),
    (['SAR_n15_v3_min-d4_pr0.50_P04.ppddl'], None),
    (['SAR_n15_v9_min-d4_pr0.25_P05.ppddl'], None),
    (['SAR_n17_v6_min-d4_pr0.75_P24.ppddl'], None),
    (['SAR_n18_v2_min-d4_pr0.50_P26.ppddl'], None),
    (['SAR_n18_v7_min-d4_pr0.75_P17.ppddl'], None),
    (['SAR_n20_v4_min-d4_pr0.10_P11.ppddl'], None),
]  # yapf: disable
