"""Unit tests for the custom ASNet ops, including finite-difference-based
gradient tests."""

import numpy as np
import pytest
import tensorflow as tf

from asnets.ops.asnet_ops import multi_gather_concat, multi_pool_concat, \
    _ref_impl_multi_gather_concat, _ref_impl_multi_pool_concat


def test_gather_op_manual():
    with tf.Session(graph=tf.Graph()):
        # start with batch size of 1
        inputs = [
            tf.constant([[
                [11.0, 12.0],
                [13.0, 14.0],
                [15.0, 16.0],
            ]]),
            tf.constant([[
                [21.3, 22.3, 23.3],
                [24.3, 25.3, 26.3],
            ]])
        ]
        indices = [
            tf.constant([2, 0, 1], dtype=tf.int64),
            tf.constant([1, 1, 0], dtype=tf.int64)
        ]
        manual_ref_out = np.array([[
            [15.0, 16.0, 24.3, 25.3, 26.3],
            [11.0, 12.0, 24.3, 25.3, 26.3],
            [13.0, 14.0, 21.3, 22.3, 23.3],
        ]])
        out_op = multi_gather_concat(inputs, indices)
        assert out_op.shape.as_list() == [1, 3, 5]
        out_vals = out_op.eval()

        # check for consistency with what I think it should be
        assert np.allclose(out_vals, manual_ref_out)

        # check for reference implementation consistency
        ref_out = _ref_impl_multi_gather_concat(inputs, indices).eval()
        assert np.allclose(out_vals, ref_out)


@pytest.mark.parametrize(
    "seed,batch_size,out_w,clist,in_w_list",
    [
        # this is here b/c it tests that grads are accumulated instead of
        # copied in-place
        (7485, 1, 2, [1], [1]),
        # an "easy" case: just tests how things are moved across
        (1293, 1, 1, [3, 9, 2, 1], [6, 8, 1, 9]),
        # test how we deal w/ bigger computations
        (2098, 16, 9, [5, 13, 6], [4, 9, 2]),
        # big computations & 0-d channels
        (6126, 64, 13, [4, 0, 18, 128], [16, 32, 3, 8]),
        # some edge cases
        (8749, 1, 8, [0], [12]),
        (9810, 0, 0, [0], [0]),
    ])
def test_gather_op_auto(seed, batch_size, out_w, clist, in_w_list):
    """Create some auto-generated matrices & index arrays to verify that they
    work fine by checking against reference impl.

    Args:
        seed (int): random seed for generating indices/values.
        batch_size (int): size of input value batch.
        out_w (int): width of output vector (number of actions, I guess).
        clist ([int]): channel counts for each input tensor.
        in_w_list ([int]): width of each input list (like number of input
            propositions in previous layer). Should be same length as clist,
            since both lists refer to the same list of input tensors!"""
    with tf.Session(graph=tf.Graph()):
        N = len(clist)
        assert N == len(in_w_list)

        # set up input data
        rng = np.random.RandomState(seed)
        inputs = [
            rng.randn(batch_size, in_w, chan_count)
            for in_w, chan_count in zip(in_w_list, clist)
        ]
        indices = [rng.randint(in_w, size=(out_w, )) for in_w in in_w_list]

        # input placeholders to feed data
        input_placeholders = tuple(
            tf.placeholder(dtype=tf.float32, shape=(None, in_w, chan_count))
            for in_w, chan_count in zip(in_w_list, clist))
        index_placeholders = tuple(
            tf.placeholder(dtype=tf.int64, shape=(out_w, )) for _ in in_w_list)

        # feed dict mapping placeholders to test values
        feed_dict = {
            input_ph: input_value
            for input_value, input_ph in zip(inputs, input_placeholders)
        }
        feed_dict.update({
            index_ph: index_value
            for index_value, index_ph in zip(indices, index_placeholders)
        })

        # output nodes in TF graph
        impl_node = multi_gather_concat(input_placeholders, index_placeholders)
        ref_node = _ref_impl_multi_gather_concat(input_placeholders,
                                                 index_placeholders)

        _do_ref_checks_auto(impl_node, ref_node, feed_dict, input_placeholders)


@pytest.mark.parametrize("oob_elem", [3, -1, 12, -99999, 999999])
def test_all_op_out_of_bounds(oob_elem):
    with tf.Session(graph=tf.Graph()):
        inputs = [tf.constant([[[1.0, 2.0, 3.0]] * 3])]
        indices_safe = [tf.constant([0, 1, 2], dtype=tf.int64)]
        pool_indices_safe = [
            tf.cast(tf.RaggedTensor.from_row_lengths(indices_safe[0], [3]),
                    tf.int64)
        ]
        indices_unsafe = [tf.constant([0, 1, oob_elem, 2], dtype=tf.int64)]
        pool_indices_unsafe = [
            tf.cast(tf.RaggedTensor.from_row_lengths(indices_unsafe[0], [4]),
                    tf.int64)
        ]
        indices_bad_type = [tf.constant([0, 1, oob_elem, 2], dtype=tf.int32)]
        pool_indices_bad_type = [
            tf.cast(tf.RaggedTensor.from_row_lengths(indices_bad_type[0], [4]),
                    tf.int32)
        ]
        min_value = 0.0
        # these two should not raise exceptions
        multi_gather_concat(inputs, indices_safe).eval()
        multi_pool_concat(inputs, pool_indices_safe, min_value).eval()
        # the next two should raise exceptions
        with pytest.raises(tf.errors.InvalidArgumentError):
            multi_gather_concat(inputs, indices_unsafe).eval()
        with pytest.raises(tf.errors.InvalidArgumentError):
            multi_pool_concat(inputs, pool_indices_unsafe, min_value).eval()
        with pytest.raises(TypeError):
            multi_gather_concat(inputs, indices_bad_type).eval()
        with pytest.raises(TypeError):
            multi_pool_concat(inputs, pool_indices_bad_type, min_value).eval()


def test_pool_op_manual():
    """Simple manual test for pool op to make sure it's doing what I expect."""
    # set up some hand-chosen inputs
    with tf.Session(graph=tf.Graph()):
        min_value = tf.constant(-1.0, dtype=tf.float32)
        pools_basic = [
            [0],  # should just copy first elem
            [],  # empty, so will get replaced with min_value
            [1, 2],  # will actually take a max
            [],  # empty; putting on end might expose segmenting bugs
        ]
        pools_values = sum(pools_basic, [])
        pools_row_lens = [len(p) for p in pools_basic]
        # No dtype arg in constructor? Weird.
        print(pools_values, pools_row_lens)
        pools = tf.RaggedTensor.from_row_lengths(pools_values, pools_row_lens)
        pools = tf.cast(pools, tf.int64)
        inputs = tf.constant(
            [
                [  # batch 1: actually exhibits interesting behaviour
                    [-2.0, 3.0],
                    [0.0, 1.0],
                    [-1.0, 2.0],
                ],
                [  # batch 2: just copies or does cutoff
                    [7.0, 7.0],
                    [7.0, 7.0],
                    [7.0, 7.0],
                ]
            ],
            dtype=tf.float32)
        expect_out = np.array(
            [
                [  # batch 1:
                    [-1.0, 3.0],
                    [-1.0, -1.0],
                    [0.0, 2.0],
                    [-1.0, -1.0],
                ],
                [  # batch 2:
                    [7.0, 7.0],
                    [-1.0, -1.0],
                    [7.0, 7.0],
                    [-1.0, -1.0],
                ]
            ],
            dtype='float32')

        # test that ref impl works (otherwise our auto tests are useless!)
        ref_out_node = _ref_impl_multi_pool_concat([inputs], [pools],
                                                   min_value)
        ref_out = ref_out_node.eval()
        assert np.all(ref_out == expect_out)

        # check that custom impl works
        custom_out_node = multi_pool_concat([inputs], [pools], min_value)
        custom_out = custom_out_node.eval()
        assert np.all(custom_out == expect_out)


def _do_ref_checks_auto(impl_node,
                        ref_node,
                        feed_dict,
                        input_placeholders,
                        *,
                        check_grads=True):
    """Do automated checks against reference implementation. Verifies both that
    results yielded by the two nodes are correct, and that gradients of
    input_placeholders w.r.t a simple squared sum loss are accurate."""
    # check shape inference
    assert impl_node.shape.as_list() == ref_node.shape.as_list()

    # check output values
    impl_out = impl_node.eval(feed_dict=feed_dict)
    ref_out = impl_node.eval(feed_dict=feed_dict)
    assert np.allclose(impl_out, ref_out)

    if check_grads:
        # check gradients
        sess = tf.get_default_session()
        impl_sqr_node = tf.reduce_sum(tf.square(impl_node))
        ref_sqr_node = tf.reduce_sum(tf.square(ref_node))
        impl_grad_node = tf.gradients(impl_sqr_node, input_placeholders)
        ref_grad_node = tf.gradients(ref_sqr_node, input_placeholders)
        impl_grad_vals = sess.run(impl_grad_node, feed_dict=feed_dict)
        ref_grad_vals = sess.run(ref_grad_node, feed_dict=feed_dict)
        assert len(impl_grad_vals) == len(ref_grad_vals)
        for impl_grad, ref_grad in zip(impl_grad_vals, ref_grad_vals):
            assert np.allclose(impl_grad, ref_grad)


@pytest.mark.parametrize(
    "seed,batch_size,out_pool_sizes,c_list,in_w_list",
    [
        # TODO: make sure this test data actually makes sense (what do c_list
        # and in_w_list mean again?)
        # another easy case that just copies things, like test for gathering op
        (1955, 1, [[1]] * 4, [3, 9, 2, 1], [6, 8, 1, 9]),
        # edge case of no channels (?!), which should lead to empty tensor
        (1746, 1, [[3, 2, 1]], [0], [12]),
        # no input channels, no propositions, no pools---we'll just get an
        # empty tensor back out
        (9528, 0, [[0]], [0], [0]),
        # test how we deal w/ bigger computations
        (6233, 16, [[1, 2], [0, 1], [2, 3]], [5, 13, 6], [4, 9, 2]),
        # big computations etc.
        (2146, 64, [[3, 2], [1, 0], [4, 4], [1, 2]], [4, 0, 18, 128], \
         [16, 32, 3, 8]),
    ])
def test_pool_op_auto(seed, batch_size, out_pool_sizes, c_list, in_w_list):
    """Test of pool op w/ auto-generated data, like test_gather_op_auto.

    Args:
        out_pool_sizes ([[int]]): nested list in which each inner list
            indicates max number of elements of corresponding input tensor to
            pool over for a series of output channel groups. e.g. [1, 3] will
            produce an output in which the result of pooling over up to one
            randomly-chosen output is stacked with the output of pooling over
            up to three randomly-chosen outputs.
        seed, batch_size, c_list, in_w_list: same as test_gather_op_auto."""
    with tf.Session(graph=tf.Graph()):
        N = len(c_list)
        assert N == len(in_w_list)
        assert N == len(out_pool_sizes)
        # all pools must be the same width, annoyingly
        assert len(set(map(len, out_pool_sizes))) == 1

        # Set up input data. min_value & ragged_pools are going to be (or
        # contain) normal TF data, whereas `inputs` will contain NP arrays that
        # get corresponding placeholder
        rng = np.random.RandomState(seed)
        min_value = tf.constant(rng.randn())
        inputs = []
        input_placeholders = []
        ragged_pools = []
        for in_w, chan_count, pool_sizes in zip(in_w_list, c_list,
                                                out_pool_sizes):
            inputs.append(rng.randn(batch_size, in_w, chan_count))
            input_placeholders.append(
                tf.placeholder(tf.float32, (None, in_w, chan_count)))
            pools_nested = [rng.permutation(in_w)[:s] for s in pool_sizes]
            pool_vals = np.concatenate(pools_nested)
            # computing this separately to pool_sizes b/c it might differ if
            # pool_sizes[i]<in_w for any element of pool_sizes
            pool_lengths = [len(p) for p in pools_nested]
            ragged_pool = tf.cast(
                tf.RaggedTensor.from_row_lengths(pool_vals, pool_lengths),
                tf.int64)
            ragged_pools.append(ragged_pool)

        feed_dict = {
            input_ph: input_value
            for input_value, input_ph in zip(inputs, input_placeholders)
        }

        impl_node = multi_pool_concat(input_placeholders, ragged_pools,
                                      min_value)
        ref_node = _ref_impl_multi_pool_concat(input_placeholders,
                                               ragged_pools, min_value)

        _do_ref_checks_auto(impl_node,
                            ref_node,
                            feed_dict,
                            input_placeholders,
                            check_grads=True)
