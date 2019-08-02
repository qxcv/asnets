"""Python wrappers for custom ASNet ops that I've re-implemented in C++ for
performance reasons."""

import os.path as osp

import tensorflow as tf

module_dir = osp.dirname(osp.abspath(__file__))
_asnet_ops = tf.load_op_library(osp.join(module_dir, '_asnet_ops_impl.so'))

__all__ = ['multi_gather_concat']


def multi_gather_concat(inputs, elem_indices, name=None):
    """Pick elements out of a selection of arrays and concatenate them
    together, like a series of gathers on different tensors followed by a
    concat along the channel axis. See _ref_impl_multi_gather_concat for the
    non-fused version.

    This fused operation offers a faster way of constructing the action module
    for an action schema, where you need to pick activation vectors out of a
    list of action tensors produced by the previous proposition layer, where
    each input action tensor corresponds to a single predicate.

    Args:
        inputs ([`B*Pi*Ci` float32 tensor]): list of N 3D float tensors. The
            ith tensor has batch size B, "width" Pi (like number of
            propositions in last layer), and channel count Ci. Pi and Ci might
            differ for different tensors in the list.
        elem_indices ([int64 tensor, shape `A`]): list of N 1D integer tensors.
            The ith tensor contains indices into the second axis of inputs[i].
            Hence, all the elements of elem_indices[i] should be in [0, Pi). If
            this is not the case, then a TypeError will be thrown (except in
            corner cases like Ci=0).
        name (str or None): optional name for the op.

    Returns:
        `B*A*(ΣCi)` float32 tensor: stacked inputs; B is original batch size
        from inputs, A is width of index arrays (i.e number of actions for this
        action schema), ΣCi is sum of channel counts for input tensors. The
        contents of `output[b,a]` will be constructed by selecting
        `inputs[0][b, elem_indices[0][a]]`, `inputs[1][b, elem_indices[1][a]]`,
        etc. and concatenating those Ci-dimensional representations
        together."""
    assert len(inputs) == len(elem_indices), \
        "inputs and elem_indcies should be op lists of same length"
    with tf.name_scope(name or 'multi_gather_concat'):
        return _asnet_ops.multi_gather_concat(inputs, elem_indices)


def _ref_impl_multi_gather_concat(inputs, elem_indices):
    """Reference implementation of multi_gather_concat. Not to be used except
    for testing and _maybe_ performance comparisons."""
    assert len(inputs) == len(elem_indices)
    with tf.name_scope('ref_impl_multi_gather_concat'):
        output_N = []
        for input_tensor, elems in zip(inputs, elem_indices):
            this_out = tf.gather(input_tensor, elems, axis=1)
            output_N.append(this_out)
        return tf.concat(output_N, axis=2)


@tf.RegisterGradient("MultiGatherConcat")
def _multi_gather_concat_grad(op, grad):
    """Gradient implementation for multi_gather_concat. Should not be
    user-visible, in general."""
    N = len(op.inputs) // 2
    assert len(op.inputs) == N * 2
    # FIXME: really this only needs the SHAPES of the original inputs, not the
    # full inputs themselves. I should consider passing in just those so that
    # TF can free the inputs during the backwards pass if it's smart enough to
    # do that (honestly, it's probably not…). (same comment is in
    # _asnets_ops_impl.cc)
    orig_inputs = op.inputs[:N]
    elem_indices = op.inputs[N:]
    inputs_grads = _asnet_ops.multi_gather_concat_grad(grad, orig_inputs,
                                                       elem_indices)
    # gradients exist only w.r.t inputs, not w.r.t elem_indices
    return list(inputs_grads) + [None] * N


def multi_pool_concat(inputs, elem_indices_ragged, min_value, name=None):
    """A version of multi_gather_concat that produces each output sub-vector
    (along last axis) by max-pooling over several vectors from an input
    tensor. This uses a fused operation under the hood, and so can serve as a
    very memory-efficient (and perhaps also fast) way of implementing
    proposition modules.

    Args:
        inputs ([`B*Ai*Ci` float32 tensor]): list of N 3D float tensors. The
            ith tensor has batch size B, "width" Ai (like number of
            actions in last layer), and channel count Ci (possibly different
            for each element of `inputs`). Basically the same as `inputs` in
            `multi_gather_concat`.
        elem_indices ([int64 ragged tensor, shape `P*?`]): list of N 2D ragged
            integer tensors. The ith tensor contains indices into the second
            axis of inputs[i], so same domain restrictions apply as for
            `multi_gather_concat` (specifically, that values must be in [0,
            Pi)).
        min_value (float): starting value to use when doing a max reduction (as
            described below). Depending on the application, this probably needs
            to be set lower than any feasible value of `inputs` (e.g -1 if
            you're using elu activation to produce `inputs`, or 0 if you're
            using relu). See below for semantics.
        name (str or None): optional name for the op.

    Returns:
        `B*P*(ΣCi)` float32 tensor: stacked inputs; B is original batch size
        from inputs, P is width of index arrays (i.e number of propositions for
        this predicate), ΣCi is sum of channel counts for input tensors. The
        contents of `output[b,p]` will be constructed by max pooling over
        `inputs[0][b, elem_indices[0][p]]`, max pooling over `inputs[1][b,
        elem_indices[1][p]]`, etc. and concatenating those Ci-dimensional
        representations together. Note that the min element produced by max
        pooling will be equal to `min_value`. A consequence of this is that if
        `elem_indices[?][p]` is an empty row, then the corresponding output
        will have value `min_value` (because there is nothing to pool over)."""
    assert len(inputs) == len(elem_indices_ragged)
    with tf.name_scope(name or 'multi_gather_concat'):
        inds_value_list = []
        inds_split_list = []
        for ragged_inds in elem_indices_ragged:
            # TODO: handle case where we're given plain tensors instead of
            # RaggedTensors. Those should just be converted to "trivial"
            # RaggedTensors. Actually, if it's a plain tensor with length >=1
            # along last axis, then I could dispatch out to
            # multi_gather_concat.
            inds_value_list.append(ragged_inds.values)
            inds_split_list.append(ragged_inds.row_splits)
        return _asnet_ops.multi_pool_concat(inputs, inds_value_list,
                                            inds_split_list, min_value)


def _ref_impl_multi_pool_concat(inputs, all_elem_indices_ragged, min_value):
    """Reference implementation of multi_pool_concat. For testing purposes
    only."""
    assert len(inputs) == len(all_elem_indices_ragged)
    results = []
    with tf.name_scope('ref_impl_multi_pool_concat'):
        for input_tens, ragged_inds in zip(inputs, all_elem_indices_ragged):
            # make sure that ragged_inds is a 2D ragged tensor, with first
            # dimension uniform and second dimension ragged
            assert len(ragged_inds.shape) == 2, ragged_inds.shape
            assert ragged_inds.ragged_rank == 1, ragged_inds.ragged_rank
            # make sure input_tens is of (static) rank 3
            assert len(input_tens.shape) == 3, input_tens.shape

            # add some padding to the end of the input tensor with min_value
            it_shape = tf.shape(input_tens)
            batch_size = it_shape[0]
            input_width = it_shape[1]
            chans = it_shape[2]
            min_pad_untiled = tf.reshape(
                tf.convert_to_tensor(min_value, dtype=tf.float32), (1, 1, 1))
            min_pad = tf.tile(min_pad_untiled, (batch_size, 1, chans))
            input_tens = tf.concat([input_tens, min_pad], axis=1)

            # transpose input_tens so that the "width" axis that we select on
            # appears first
            input_tens_trans = tf.transpose(input_tens, (1, 0, 2))

            # also extend elem_indices with a ref to the padding element so
            # that max pooling yields result bounded below by min_value
            min_pad_ind_untiled = tf.cast(tf.reshape(input_width, (1, 1)),
                                          tf.int64)
            out_width = ragged_inds.nrows()
            min_pad_tiled = tf.tile(min_pad_ind_untiled, (out_width, 1))
            ragged_inds = tf.cast(ragged_inds, tf.int64)
            ragged_inds = tf.concat([ragged_inds, min_pad_tiled], axis=1)

            # implement the max pool with a max reduction on a gathered tensor
            gathered_tens_trans = tf.gather(input_tens_trans,
                                            ragged_inds,
                                            axis=0)
            reduced_tens_trans = tf.reduce_max(gathered_tens_trans, axis=1)

            # transpose back to original format & add to results (w/ assert to
            # make sure output shape is right!)
            reduced_tens = tf.transpose(reduced_tens_trans, (1, 0, 2))
            expected_shape = tf.stack(
                [batch_size, tf.cast(out_width, tf.int32), chans], axis=0)
            eq_shape_assert = tf.assert_equal(tf.shape(reduced_tens),
                                              expected_shape)
            with tf.control_dependencies([eq_shape_assert]):
                reduced_tens = tf.identity(reduced_tens)
            results.append(reduced_tens)

        # concatenate along channels
        final_output = tf.concat(results, axis=2)

    return final_output


@tf.RegisterGradient("MultiPoolConcat")
def _multi_pool_concat_grad(op, grad):
    """Gradient impl for multi_pool_concat."""
    N = (len(op.inputs) - 1) // 3
    assert len(op.inputs) == 3 * N + 1
    orig_inputs = op.inputs[:N]
    orig_elem_inds_vals = op.inputs[N:2*N]
    orig_elem_inds_splits = op.inputs[2*N:3*N]
    orig_output, = op.outputs
    input_grads = _asnet_ops.multi_pool_concat_grad(grad, orig_inputs,
                                                    orig_elem_inds_vals,
                                                    orig_elem_inds_splits,
                                                    orig_output)
    return list(input_grads) + [None] * (2 * N + 1)
