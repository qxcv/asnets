"""For experiments on a probabilistic blocksworld domain like the one from
IPPC'08 (probabilistically uninteresting one, not exploding blocksworld)."""

PDDL_DIR = '../problems/mine/prob-bw-redux/'
COMMON_PDDLS = ['prob-bw.pddl']
TRAIN_PDDLS = [
    # size 4
    # 'problems/prob_bw_n4_es1.pddl',
    # 'problems/prob_bw_n4_es2.pddl',
    # 'problems/prob_bw_n4_es3.pddl',
    # 'problems/prob_bw_n4_es4.pddl',
    # 'problems/prob_bw_n4_es5.pddl',
    # size 5
    'problems/prob_bw_n5_es1.pddl',
    'problems/prob_bw_n5_es2.pddl',
    'problems/prob_bw_n5_es3.pddl',
    'problems/prob_bw_n5_es4.pddl',
    'problems/prob_bw_n5_es5.pddl',
    # 'problems/prob_bw_n5_es6.pddl',
    # 'problems/prob_bw_n5_es7.pddl',
    # 'problems/prob_bw_n5_es8.pddl',
    # 'problems/prob_bw_n5_es9.pddl',
    # 'problems/prob_bw_n5_es10.pddl',
    # size 6
    'problems/prob_bw_n6_es1.pddl',
    'problems/prob_bw_n6_es2.pddl',
    'problems/prob_bw_n6_es3.pddl',
    'problems/prob_bw_n6_es4.pddl',
    'problems/prob_bw_n6_es5.pddl',
    # 'problems/prob_bw_n6_es6.pddl',
    # 'problems/prob_bw_n6_es7.pddl',
    # 'problems/prob_bw_n6_es8.pddl',
    # 'problems/prob_bw_n6_es9.pddl',
    # 'problems/prob_bw_n6_es10.pddl',
    # size 7
    'problems/prob_bw_n7_es1.pddl',
    'problems/prob_bw_n7_es2.pddl',
    'problems/prob_bw_n7_es3.pddl',
    'problems/prob_bw_n7_es4.pddl',
    'problems/prob_bw_n7_es5.pddl',
    # size 8
    'problems/prob_bw_n8_es1.pddl',
    'problems/prob_bw_n8_es2.pddl',
    'problems/prob_bw_n8_es3.pddl',
    'problems/prob_bw_n8_es4.pddl',
    'problems/prob_bw_n8_es5.pddl',
    # size 9
    'problems/prob_bw_n9_es1.pddl',
    'problems/prob_bw_n9_es2.pddl',
    'problems/prob_bw_n9_es3.pddl',
    'problems/prob_bw_n9_es4.pddl',
    'problems/prob_bw_n9_es5.pddl',
]  # yapf: disable
TRAIN_NAMES = None
TEST_RUNS = [
    # This is a lot of instances. I ended up dropping all but 30 of the largest
    # instances, all randomly generated. Note that only a small subset of these
    # appeared in the original AAAI paper (look for 'problems-aaai-orig')
    # # size 4 (orig set)
    # (['problems-aaai-orig/prob_bw_n4_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n4_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n4_s3.pddl'], None),
    # # size 5 (orig set)
    # (['problems-aaai-orig/prob_bw_n5_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n5_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n5_s3.pddl'], None),
    # # size 6 (orig set)
    # (['problems-aaai-orig/prob_bw_n6_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n6_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n6_s3.pddl'], None),
    # # size 7 (orig set)
    # (['problems-aaai-orig/prob_bw_n7_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n7_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n7_s3.pddl'], None),
    # # size 8 (orig set)
    # (['problems-aaai-orig/prob_bw_n8_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n8_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n8_s3.pddl'], None),
    # # size 9 (orig set)
    # (['problems-aaai-orig/prob_bw_n9_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n9_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n9_s3.pddl'], None),
    # # size 10 (orig set)
    # (['problems-aaai-orig/prob_bw_n10_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n10_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n10_s3.pddl'], None),
    # # size 10 (big set)
    # (['problems/prob_bw_n10_es1.pddl'], None),
    # (['problems/prob_bw_n10_es2.pddl'], None),
    # (['problems/prob_bw_n10_es3.pddl'], None),
    # (['problems/prob_bw_n10_es4.pddl'], None),
    # (['problems/prob_bw_n10_es5.pddl'], None),
    # (['problems/prob_bw_n10_es6.pddl'], None),
    # (['problems/prob_bw_n10_es7.pddl'], None),
    # (['problems/prob_bw_n10_es8.pddl'], None),
    # (['problems/prob_bw_n10_es9.pddl'], None),
    # (['problems/prob_bw_n10_es10.pddl'], None),
    # size 15 (orig set)
    # (['problems-aaai-orig/prob_bw_n15_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n15_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n15_s3.pddl'], None),
    # size 15 (big set)
    (['problems/prob_bw_n15_es1.pddl'], None),
    (['problems/prob_bw_n15_es2.pddl'], None),
    (['problems/prob_bw_n15_es3.pddl'], None),
    (['problems/prob_bw_n15_es4.pddl'], None),
    (['problems/prob_bw_n15_es5.pddl'], None),
    # (['problems/prob_bw_n15_es6.pddl'], None),
    # (['problems/prob_bw_n15_es7.pddl'], None),
    # (['problems/prob_bw_n15_es8.pddl'], None),
    # (['problems/prob_bw_n15_es9.pddl'], None),
    # (['problems/prob_bw_n15_es10.pddl'], None),
    # size 20 (orig set)
    # (['problems-aaai-orig/prob_bw_n20_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n20_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n20_s3.pddl'], None),
    # size 20 (big set)
    (['problems/prob_bw_n20_es1.pddl'], None),
    (['problems/prob_bw_n20_es2.pddl'], None),
    (['problems/prob_bw_n20_es3.pddl'], None),
    (['problems/prob_bw_n20_es4.pddl'], None),
    (['problems/prob_bw_n20_es5.pddl'], None),
    # (['problems/prob_bw_n20_es6.pddl'], None),
    # (['problems/prob_bw_n20_es7.pddl'], None),
    # (['problems/prob_bw_n20_es8.pddl'], None),
    # (['problems/prob_bw_n20_es9.pddl'], None),
    # (['problems/prob_bw_n20_es10.pddl'], None),
    # size 25 (orig set)
    # (['problems-aaai-orig/prob_bw_n25_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n25_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n25_s3.pddl'], None),
    # size 25 (big set)
    (['problems/prob_bw_n25_es1.pddl'], None),
    (['problems/prob_bw_n25_es2.pddl'], None),
    (['problems/prob_bw_n25_es3.pddl'], None),
    (['problems/prob_bw_n25_es4.pddl'], None),
    (['problems/prob_bw_n25_es5.pddl'], None),
    # (['problems/prob_bw_n25_es6.pddl'], None),
    # (['problems/prob_bw_n25_es7.pddl'], None),
    # (['problems/prob_bw_n25_es8.pddl'], None),
    # (['problems/prob_bw_n25_es9.pddl'], None),
    # (['problems/prob_bw_n25_es10.pddl'], None),
    # size 30 (orig set)
    # (['problems-aaai-orig/prob_bw_n30_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n30_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n30_s3.pddl'], None),
    # size 30 (big set)
    (['problems/prob_bw_n30_es1.pddl'], None),
    (['problems/prob_bw_n30_es2.pddl'], None),
    (['problems/prob_bw_n30_es3.pddl'], None),
    (['problems/prob_bw_n30_es4.pddl'], None),
    (['problems/prob_bw_n30_es5.pddl'], None),
    # (['problems/prob_bw_n30_es6.pddl'], None),
    # (['problems/prob_bw_n30_es7.pddl'], None),
    # (['problems/prob_bw_n30_es8.pddl'], None),
    # (['problems/prob_bw_n30_es9.pddl'], None),
    # (['problems/prob_bw_n30_es10.pddl'], None),
    # size 35 (orig set)
    # (['problems-aaai-orig/prob_bw_n35_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n35_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n35_s3.pddl'], None),
    # size 35 (big set)
    (['problems/prob_bw_n35_es1.pddl'], None),
    (['problems/prob_bw_n35_es2.pddl'], None),
    (['problems/prob_bw_n35_es3.pddl'], None),
    (['problems/prob_bw_n35_es4.pddl'], None),
    (['problems/prob_bw_n35_es5.pddl'], None),
    # (['problems/prob_bw_n35_es6.pddl'], None),
    # (['problems/prob_bw_n35_es7.pddl'], None),
    # (['problems/prob_bw_n35_es8.pddl'], None),
    # (['problems/prob_bw_n35_es9.pddl'], None),
    # (['problems/prob_bw_n35_es10.pddl'], None),
    # size 40 (orig set)
    # (['problems-aaai-orig/prob_bw_n40_s1.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n40_s2.pddl'], None),
    # (['problems-aaai-orig/prob_bw_n40_s3.pddl'], None),
    # size 40 (big set)
    (['problems/prob_bw_n40_es1.pddl'], None),
    (['problems/prob_bw_n40_es2.pddl'], None),
    (['problems/prob_bw_n40_es3.pddl'], None),
    (['problems/prob_bw_n40_es4.pddl'], None),
    (['problems/prob_bw_n40_es5.pddl'], None),
    # (['problems/prob_bw_n40_es6.pddl'], None),
    # (['problems/prob_bw_n40_es7.pddl'], None),
    # (['problems/prob_bw_n40_es8.pddl'], None),
    # (['problems/prob_bw_n40_es9.pddl'], None),
    # (['problems/prob_bw_n40_es10.pddl'], None),
]  # yapf: disable
