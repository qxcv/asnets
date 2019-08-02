"""For experiments on Exploding Blocksworld."""

PDDL_DIR = '../problems/mine/ex-bw/'
COMMON_PDDLS = ['domain.pddl']
TRAIN_PDDLS = [
    # Sam: I've uncommented things for which reasonable plans are findable in
    # roughly 2s (excluding the four-block problems, which are probably too
    # easy)
    # 'train/ex-bw-train-n04-s00-r490007.pddl',
    # 'train/ex-bw-train-n04-s01-r339112.pddl',
    # 'train/ex-bw-train-n04-s02-r506078.pddl',
    # 'train/ex-bw-train-n04-s03-r692020.pddl',
    # 'train/ex-bw-train-n04-s04-r540416.pddl',
    # 'train/ex-bw-train-n04-s05-r256409.pddl',
    # 'train/ex-bw-train-n04-s06-r879698.pddl',
    # 'train/ex-bw-train-n04-s07-r331801.pddl',
    # 'train/ex-bw-train-n04-s08-r270660.pddl',
    # 'train/ex-bw-train-n04-s09-r768007.pddl',

    # V(s0)=8
    'train/ex-bw-train-n05-s00-r275246.pddl',
    #
    # V(s0)=2 (probably trivial; I think I have a bug where I don't check goal
    # condition at beginning)
    # 'train/ex-bw-train-n05-s01-r874185.pddl',
    #
    # V(s0)=8
    'train/ex-bw-train-n05-s02-r103966.pddl',
    #
    # V(s0)=8
    'train/ex-bw-train-n05-s03-r422575.pddl',
    #
    # V(s0)=???
    # 'train/ex-bw-train-n05-s04-r658439.pddl',
    #
    # V(s0)=8
    'train/ex-bw-train-n05-s05-r427462.pddl',
    #
    # V(s0)=8
    'train/ex-bw-train-n05-s06-r181426.pddl',
    #
    # V(s0)~=28
    'train/ex-bw-train-n05-s07-r318797.pddl',
    #
    # 'train/ex-bw-train-n05-s08-r101747.pddl',
    # V(s0)~=112
    'train/ex-bw-train-n05-s09-r820796.pddl',

    # 'train/ex-bw-train-n06-s00-r925408.pddl',
    #
    # V(s0)~=150.88
    'train/ex-bw-train-n06-s01-r803196.pddl',
    #
    # 'train/ex-bw-train-n06-s02-r417719.pddl',
    #
    # V(s0)=10
    'train/ex-bw-train-n06-s03-r193620.pddl',
    #
    # 'train/ex-bw-train-n06-s04-r132017.pddl',
    #
    # 'train/ex-bw-train-n06-s05-r100631.pddl',
    #
    # V(s0)~=53.6
    'train/ex-bw-train-n06-s06-r227402.pddl',
    #
    # 'train/ex-bw-train-n06-s07-r320190.pddl',
    #
    # 'train/ex-bw-train-n06-s08-r241599.pddl',
    #
    # 'train/ex-bw-train-n06-s09-r859459.pddl',

    # V(s0)~=12
    'train/ex-bw-train-n07-s00-r103415.pddl',
    #
    # 'train/ex-bw-train-n07-s01-r456211.pddl',
    #
    # V(s0)~=361.1
    'train/ex-bw-train-n07-s02-r104033.pddl',
    #
    # 'train/ex-bw-train-n07-s03-r201931.pddl',
    #
    # V(s0)~=117.42
    'train/ex-bw-train-n07-s04-r102044.pddl',
    #
    # V(s0)~=111.11
    'train/ex-bw-train-n07-s05-r765625.pddl',
    #
    # V(s0)~=14
    'train/ex-bw-train-n07-s06-r967519.pddl',
    #
    # 'train/ex-bw-train-n07-s07-r954403.pddl',
    #
    # V(s0)~=14
    'train/ex-bw-train-n07-s08-r508090.pddl',
    #
    # 'train/ex-bw-train-n07-s09-r502923.pddl',

    # 'train/ex-bw-train-n08-s00-r846148.pddl',
    # 'train/ex-bw-train-n08-s01-r900927.pddl',
    # V(s0)~=525
    'train/ex-bw-train-n08-s02-r883580.pddl',
    # 'train/ex-bw-train-n08-s03-r558239.pddl',
    # V(s0)~=525
    'train/ex-bw-train-n08-s04-r659102.pddl',
    # V(s0)~=92.02
    'train/ex-bw-train-n08-s05-r107808.pddl',
    # V(s0)~=132.80
    'train/ex-bw-train-n08-s06-r103688.pddl',
    # 'train/ex-bw-train-n08-s07-r107722.pddl',
    # V(s0)~=525
    'train/ex-bw-train-n08-s08-r398914.pddl',
    # 'train/ex-bw-train-n08-s09-r104548.pddl',
    # V(s0)=58.47
    'train/ex-bw-train-n09-s00-r218544.pddl',
    # 'train/ex-bw-train-n09-s01-r101821.pddl',
    # 'train/ex-bw-train-n09-s02-r251832.pddl',
    #
    # V(s0)=752.31
    'train/ex-bw-train-n09-s03-r612632.pddl',
    #
    # 'train/ex-bw-train-n09-s04-r754958.pddl',
    # 'train/ex-bw-train-n09-s05-r952695.pddl',
    #
    # V(s0)=482.64
    'train/ex-bw-train-n09-s06-r537935.pddl',
    #
    # 'train/ex-bw-train-n09-s07-r209513.pddl',
    # 'train/ex-bw-train-n09-s08-r465095.pddl',
    # 'train/ex-bw-train-n09-s09-r247953.pddl',
]
TRAIN_NAMES = None
_TEST_RUNS = [
    'test/ex-bw-test-n11-s00-r972773.pddl',
    'test/ex-bw-test-n11-s01-r763719.pddl',
    'test/ex-bw-test-n11-s02-r925084.pddl',
    'test/ex-bw-test-n12-s00-r270300.pddl',
    'test/ex-bw-test-n12-s01-r846863.pddl',
    'test/ex-bw-test-n12-s02-r595497.pddl',
    'test/ex-bw-test-n13-s00-r999827.pddl',
    'test/ex-bw-test-n13-s01-r325159.pddl',
    'test/ex-bw-test-n13-s02-r948356.pddl',
    'test/ex-bw-test-n14-s00-r218782.pddl',
    'test/ex-bw-test-n14-s01-r846992.pddl',
    'test/ex-bw-test-n14-s02-r105760.pddl',
    'test/ex-bw-test-n15-s00-r717403.pddl',
    'test/ex-bw-test-n15-s01-r201725.pddl',
    'test/ex-bw-test-n15-s02-r106727.pddl',
    'test/ex-bw-test-n16-s00-r109291.pddl',
    'test/ex-bw-test-n16-s01-r253915.pddl',
    'test/ex-bw-test-n16-s02-r329368.pddl',
    'test/ex-bw-test-n17-s00-r546126.pddl',
    'test/ex-bw-test-n17-s01-r453173.pddl',
    'test/ex-bw-test-n17-s02-r107890.pddl',
    'test/ex-bw-test-n18-s00-r119378.pddl',
    'test/ex-bw-test-n18-s01-r769356.pddl',
    'test/ex-bw-test-n18-s02-r734006.pddl',
    'test/ex-bw-test-n19-s00-r491316.pddl',
    'test/ex-bw-test-n19-s01-r657239.pddl',
    'test/ex-bw-test-n19-s02-r534491.pddl',
    'test/ex-bw-test-n20-s00-r167713.pddl',
    'test/ex-bw-test-n20-s01-r196207.pddl',
    'test/ex-bw-test-n20-s02-r465140.pddl',
]  # yapf: disable
TEST_RUNS = [([name], None) for name in _TEST_RUNS]
