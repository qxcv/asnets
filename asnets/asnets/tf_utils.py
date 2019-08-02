"""Generic tools for use with TensorFlow"""

import re
import numpy as np
import tensorflow as tf


def broadcast_to(pattern, array):
    """Broacast ``array`` to match shape of ``pattern``. Does not really follow
    normal broadcasting rules. Basically, array and pattern have to be of same
    rank, and if it's mathematically possible to tile out array to match
    pattern in one axis then this will do so (regardless of whether the
    dimension of the array is 1 in that axis)."""
    with tf.name_scope('broadcast_to'):
        pat_shape = tf.shape(pattern)
        arr_shape = tf.shape(array)
        denom = tf.where(
            tf.equal(pat_shape, 0), tf.ones_like(arr_shape), arr_shape)
        multiples = tf.floordiv(pat_shape, denom)
        rv = tf.tile(array, multiples)
        # now check shape matches
        rv_shape = tf.shape(rv)
        shapes_equal = tf.reduce_all(tf.equal(pat_shape, rv_shape))
        shape_assert = tf.Assert(
            shapes_equal, [pat_shape, arr_shape, multiples, rv_shape],
            name='shape_assert')
        with tf.control_dependencies([shape_assert]):
            return tf.identity(rv)


def masked_softmax(activations, mask, name='masked_softmax'):
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
            activations, reduction_indices=[-1], keepdims=True)
        min_acts = broadcast_to(activations, min_acts)
        disabled_min_out = tf.where(
            mask, activations, min_acts, name='disabled_to_min')
        # subtract out maximum for numeric stability
        max_acts = tf.reduce_max(
            disabled_min_out, reduction_indices=[-1], keepdims=True)
        max_acts = broadcast_to(activations, max_acts)
        pad_acts = activations - max_acts
        # just to be safe, set to 0 where not valid (so we don't accidentally
        # take exp() of a big number corresponding to a deactivated action, as
        # may happen when max_acts is negative & really, really low but one of
        # the deactivated actions is already high)
        pad_acts = tf.where(
            mask, pad_acts, tf.zeros_like(pad_acts), name='zero_before_exp')
        exps = tf.cast(mask, tf.float32) * tf.exp(pad_acts, name='masked_exps')
        # use uniform predictions when nothing is valid
        # TODO: also replace with uniform when only one thing is valid---what's
        # the point of predicting then? probably just lead to overfitting w/o
        # driving real loss down!
        any_valid = tf.reduce_any(
            mask, reduction_indices=[-1], keepdims=True, name='any_valid')
        any_valid = broadcast_to(activations, any_valid)
        # signature: tf.where(switch expr, if true, if false)
        safe_exps = tf.where(
            any_valid, exps, tf.ones_like(exps), name='safe_exps')

        with tf.name_scope('normalise'):
            # now we can divide out by sums of exponentials
            sums = tf.reduce_sum(
                safe_exps, reduction_indices=-1, keepdims=True, name='sums')
            # this is just for safety
            clipped_sums = tf.clip_by_value(sums, 1e-5, 1e10)
            output = safe_exps / clipped_sums

    return output


def cross_entropy(act_dist, labels):
    # This operates on probabilities, not logits, using approach from
    # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L2725-L2753
    # Logits would probably be more stable but are harder to deal with in my
    # masked softmax formulation.
    # TODO: do I even need this given I'm using softmax? Should be normalised
    # already, no? Maybe replace with assertion.
    with tf.name_scope('cross_entropy'):
        sums = tf.reduce_sum(act_dist, axis=-1, name='sums')
        all_good = tf.reduce_all(
            tf.abs(sums - 1) < 1e-2, axis=None, name='all_good')
        control_data = [
            sums, tf.reduce_min(sums, name='sums'),
            tf.reduce_max(sums, name='reduce_max_sums'),
            tf.reduce_all(tf.is_finite(sums), name='all_finite_sums')
        ]
        check_normed_op = tf.Assert(
            all_good, control_data, name='check_normed')
        # act_dist /= tf.reduce_sum(act_dist, axis=-1, keepdims=True)
        with tf.control_dependencies([check_normed_op]):
            eps = 1e-8
            clipped = tf.clip_by_value(act_dist, eps, 1 - eps, name='clip_eps')
            return tf.reduce_sum(
                -labels * tf.log(clipped), axis=-1, name='xent_sum')


_tf_invalid_char_re = re.compile(r'[^A-Za-z0-9_.\-/]+')
_tf_invalid_char_re_noslash = re.compile(r'[^A-Za-z0-9_.\-]+')


def escape_name_tf(name, allow_slash=False):
    """Escape characters in a given string so that it can be used as a valid TF
    scope (and presumably op name, etc.). This uses a regex that I got from the
    source of name_scope in TF:

    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/framework/ops.py#L2993
    """
    if allow_slash:
        return _tf_invalid_char_re.sub('-', name)
    return _tf_invalid_char_re_noslash.sub('-', name)


def empty_feed_value(placeholder):
    """Takes a TF placeholder and returns an empty Numpy ndarray with the same
    shape, except on axes with size `None`, which are set to length 0. There
    should be at least one axis with a `None` or 0 length, otherwise we won't
    be able to return an empty array!"""
    ph_shape = placeholder.shape.as_list()
    new_shape = tuple(v or 0 for v in ph_shape)
    shape_prod = np.prod(new_shape)
    assert shape_prod == 0, \
        "passed tensor has shape '%s', but should have at least one 0 or " \
        "None dimension" % shape_prod
    dtype = placeholder.dtype.name
    return np.zeros(new_shape, dtype=dtype)
