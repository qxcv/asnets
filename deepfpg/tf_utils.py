"""Generic tools for use with TensorFlow"""

import tensorflow as tf
from tensorflow.python.framework import ops


def broadcast_to(pattern, array):
    """Broacast ``array`` to match shape of ``pattern``."""
    with tf.name_scope('broadcast_to'):
        pat_shape = tf.shape(pattern)
        arr_shape = tf.shape(array)
        multiples = tf.floordiv(pat_shape, arr_shape)
        pos_assert = tf.Assert(
            tf.reduce_all(multiples > 0), [multiples, pat_shape, arr_shape],
            name='pos_assert')
        with tf.control_dependencies([pos_assert]):
            rv = tf.tile(array, multiples)
            rv_shape = tf.shape(rv)
            shape_assert = tf.Assert(
                tf.reduce_all(pat_shape == rv_shape), [pat_shape, rv_shape],
                name='shape_assert')
            with tf.control_dependencies([shape_assert]):
                return rv


def masked_softmax(activations, mask):
    # elements with mask = 0 are disabled from being selected by softmax
    # (unless no other option is available)
    with tf.name_scope('masked_softmax'):
        eq_size_op = tf.assert_equal(
            tf.shape(activations),
            tf.shape(mask),
            message='activation and mask shape differ')
        with tf.control_dependencies([eq_size_op]):
            mask = tf.not_equal(mask, 0)
        # set all activations for disabled things to have minimum value
        min_acts = tf.reduce_min(
            activations, reduction_indices=[-1], keep_dims=True)
        min_acts = broadcast_to(activations, min_acts)
        disabled_min_out = tf.where(
            mask, activations, min_acts, name='disabled_to_min')
        # subtract out maximum for numeric stability
        max_acts = tf.reduce_max(
            disabled_min_out, reduction_indices=[-1], keep_dims=True)
        max_acts = broadcast_to(activations, max_acts)
        pad_acts = activations - max_acts
        exps = tf.cast(mask, tf.float32) * tf.exp(pad_acts, name='masked_exps')
        # use uniform predictions when nothing is valid
        any_valid = tf.reduce_any(
            mask, reduction_indices=[-1], keep_dims=True, name='any_valid')
        any_valid = broadcast_to(activations, any_valid)
        # signature: tf.where(switch expr, if true, if false)
        safe_exps = tf.where(
            any_valid, exps, tf.ones_like(exps), name='safe_exps')

        # now we can divide out by sums of exponentials
        sums = tf.reduce_sum(
            safe_exps, reduction_indices=-1, keep_dims=True, name='sums')
        # this is just for safety
        clipped_sums = tf.clip_by_value(sums, 1e-5, 1e10)
        output = safe_exps / clipped_sums

    return output


def make_leaky_relu(alpha=0.1):
    assert alpha <= 1, "max() impl trick won't work with big alpha"

    def inner(features):
        with tf.name_scope('leaky_relu'):
            return tf.maximum(features, alpha * features)

    return inner


def selu(x):
    """Self-normalising exponential linear unit
    (https://arxiv.org/pdf/1706.02515.pdf). Implementation taken from
    https://github.com/bioinf-jku/SNNs/blob/master/selu.py """
    with ops.name_scope('selu'):
        # XXX: this won't play nicely with dropout
        # read p.6 of the paper ("new dropout technique")
        # the repo linked above actually has an implementation of said dropout
        # technique
        assert False, "see note above about dropout; only do this once you have experiments down!"
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
