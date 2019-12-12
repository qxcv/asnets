"""Code that implements action schema network *model*. Also includes code for
an MLP. This file doesn't include training code, etc.---see other files for
that."""

from abc import ABCMeta, abstractmethod

import joblib
import tensorflow as tf
import numpy as np

from asnets.ops.asnet_ops import multi_gather_concat, multi_pool_concat
from asnets.prof_utils import can_profile
from asnets.tf_utils import masked_softmax

# should we use custom "multi_gather_concat" for action modules?
USE_CUSTOM_MULTI_GATHER_CONCAT = True
USE_CUSTOM_MULTI_POOL_CONCAT = True
# note that choice of default value for max-pooling depends on the
# non-linearity; for elu it is -1 (the minimum possible), while for relu it
# might be 0
NONLINEARITY = 'elu'


class PropNetworkWeights:
    """Manages weights for a domain-specific problem network. Those weights can
    then be used in problem-specific networks."""

    # WARNING: you need to change __{get,set}state__ if you change __init__ or
    # _make_weights()!

    def __init__(self, dom_meta, hidden_sizes, extra_dim, skip):
        # note that hidden_sizes is list of (act layer size, prop layer size)
        # pairs.
        # extra_input is just the number of extra items included in the input
        # vector for each action
        self.dom_meta = dom_meta
        self.hidden_sizes = list(hidden_sizes)
        self.extra_dim = extra_dim
        self.skip = skip
        self._make_weights()

    def __getstate__(self):
        """Pickle weights ourselves, since TF stuff is hard to pickle."""
        prop_weights_np = self._serialise_weight_list(self.prop_weights)
        act_weights_np = self._serialise_weight_list(self.act_weights)
        return {
            'dom_meta': self.dom_meta,
            'hidden_sizes': self.hidden_sizes,
            'prop_weights_np': prop_weights_np,
            'act_weights_np': act_weights_np,
            'extra_dim': self.extra_dim,
            'skip': self.skip,
        }

    def __setstate__(self, state):
        """Unpickle weights"""
        self.dom_meta = state['dom_meta']
        self.hidden_sizes = state['hidden_sizes']
        self.extra_dim = state['extra_dim']
        # old network snapshots always had skip connections turned on
        self.skip = state.get('skip', True)
        self._make_weights(state['prop_weights_np'], state['act_weights_np'])

    @staticmethod
    def _serialise_weight_list(weight_list):
        # serialises a list of dicts, each mapping str -> (tensorflow weights,
        # ...)
        sess = tf.get_default_session()
        rv = []
        for d in weight_list:
            new_d = {}
            for k, v in d.items():
                new_d[k] = tuple(sess.run(v))
            rv.append(new_d)
        return rv

    @staticmethod
    def _make_conv_weights(in_size, name_prefix, out_size, old_weights=None):
        # convenience function for making 1D convolution layer weights
        if old_weights is not None:
            W_init, b_init = map(tf.constant_initializer, old_weights)
        else:
            W_init = tf.contrib.layers.xavier_initializer_conv2d()
            b_init = tf.zeros_initializer()
        gv_kwargs = dict(dtype=tf.float32, regularizer=None, trainable=True)
        W_name = name_prefix + '/W'
        W = tf.get_variable(initializer=W_init,
                            shape=(in_size, out_size),
                            name=W_name,
                            **gv_kwargs)
        tf.summary.histogram('weights/' + W_name, W, collections=['weights'])
        b_name = name_prefix + '/b'
        b = tf.get_variable(initializer=b_init,
                            shape=(out_size, ),
                            name=b_name,
                            **gv_kwargs)
        tf.summary.histogram('weights/' + b_name, b, collections=['weights'])
        return W, b

    @can_profile
    def _make_weights(self, old_prop_weights=None, old_act_weights=None):
        # prop_weights[i] is a dictionary mapping predicate names to weights
        # for modules in the i-th proposition layer
        self.prop_weights = []
        self.act_weights = []
        self.all_weights = []

        # TODO: constructing weights separately like this (and having
        # to restore with tf.const, etc.) is silly. Should store
        # parameters *purely* by name, and have code responsible for
        # automatically re-instantiating old weights (after network
        # construction) if they exist. TF offers several ways of doing
        # exactly that.

        for hid_idx, hid_sizes in enumerate(self.hidden_sizes):
            act_size, prop_size = hid_sizes

            # make action layer weights
            act_dict = {}
            for unbound_act in self.dom_meta.unbound_acts:
                preds = self.dom_meta.rel_pred_names(unbound_act)
                if not hid_idx:
                    # first layer, so our input is actually a binary vector
                    # giving a truth value for each proposition
                    in_size = len(preds) * 2 + self.extra_dim
                else:
                    # prop inputs + skip input from previous action layer
                    in_size = len(preds) * self.hidden_sizes[hid_idx - 1][1]
                    if self.skip:
                        in_size = in_size + self.hidden_sizes[hid_idx - 1][0]

                name_pfx = 'hid_%d_act_%s' % (hid_idx, unbound_act.schema_name)
                if old_act_weights is not None:
                    act_W, act_b = self._make_conv_weights(
                        in_size,
                        name_pfx,
                        act_size,
                        old_weights=old_act_weights[hid_idx][unbound_act])
                else:
                    act_W, act_b = self._make_conv_weights(in_size,
                                                           name_pfx,
                                                           act_size,
                                                           old_weights=None)
                act_dict[unbound_act] = (act_W, act_b)
                self.all_weights.extend([act_W, act_b])

            self.act_weights.append(act_dict)

            # make hidden proposition layer weights
            pred_dict = {}
            for pred_name in self.dom_meta.pred_names:
                rel_act_slots = self.dom_meta.rel_act_slots(pred_name)
                # We should never end up with NO relevant actions & slots for a
                # predicate, else there's probably issue with domain.
                assert len(rel_act_slots) > 0, \
                    "no relevant actions for proposition %s" % pred_name

                in_size = len(rel_act_slots) * act_size
                if hid_idx and self.skip:
                    # skip connection from previous prop layer
                    in_size = in_size + self.hidden_sizes[hid_idx - 1][1]
                name_pfx = 'hid_%d_prop_%s' % (hid_idx, pred_name)
                if old_prop_weights is not None:
                    prop_W, prop_b = self._make_conv_weights(
                        in_size,
                        name_pfx,
                        prop_size,
                        old_weights=old_prop_weights[hid_idx][pred_name])
                else:
                    prop_W, prop_b = self._make_conv_weights(in_size,
                                                             name_pfx,
                                                             prop_size,
                                                             old_weights=None)
                pred_dict[pred_name] = (prop_W, prop_b)
                self.all_weights.extend([prop_W, prop_b])

            self.prop_weights.append(pred_dict)

        # make final layer weights (action)
        final_act_dict = {}
        for unbound_act in self.dom_meta.unbound_acts:
            preds = self.dom_meta.rel_pred_names(unbound_act)
            if not self.hidden_sizes:
                in_size = len(preds) * 2 + self.extra_dim
            else:
                in_size = len(preds) * self.hidden_sizes[-1][1]
                if self.skip:
                    in_size = in_size + self.hidden_sizes[-1][0]

            name_pfx = 'final_act_%s' % unbound_act.schema_name
            if old_act_weights is not None:
                final_act_W, final_act_b = self._make_conv_weights(
                    in_size,
                    name_pfx,
                    1,
                    old_weights=old_act_weights[-1][unbound_act])
            else:
                final_act_W, final_act_b = self._make_conv_weights(
                    in_size, name_pfx, 1, old_weights=None)
            final_act_dict[unbound_act] = (final_act_W, final_act_b)
            self.all_weights.extend([final_act_W, final_act_b])

        self.act_weights.append(final_act_dict)

    @can_profile
    def save(self, path):
        """Save a snapshot of the current network weights to the given path."""
        joblib.dump(self, path, compress=True)


def _check_index_spec(index_spec):
    """Sanity-check the pooling specification for a pick-pool-and-stack
    layer."""
    # shape of index_spec: [(int, [[int]])]
    # each element of index_spec is pair of (index into inputs, [[indices
    # into given input]]) Note that I have a nested list ([[int]]) because
    # each output could be *pooled* across multiple inputs.
    out_len = None

    for chosen_input, pools in index_spec:
        assert isinstance(chosen_input, int), \
            "input selector must be integer"

        for pool in pools:
            # I had a length check in before, but then realised that pools
            # could be empty :/
            # assert len(pool) >= 0
            assert all(isinstance(idx, int) for idx in pool), \
                "indices in pool must be integers"

        if out_len is None:
            out_len = len(pools)
        else:
            assert out_len == len(pools), \
                "need same pool count from each input, but %d != %d" \
                % (out_len, len(pools))


@can_profile
def pick_pool_and_stack(inputs,
                        index_spec,
                        name='pps',
                        debug=False,
                        extra_chans=None):
    """Takes some previous layers and some indices, and creates a new data
    layer by picking out the appropriate items then concatenating and/ or
    pooling them together.

    Yes, that description is extremely vague. In my defense, this layer is an
    important part of my implementation strategy for action/proposition
    modules, so it has to do a lot of unrelated things. Hopefully I'll be able
    to write a better explanation later."""

    assert len(index_spec) > 0, "need at least one incoming"
    _check_index_spec(index_spec)

    with tf.name_scope(name):
        # out_stacks will be an array of len(self.index_spec) tensors, each N *
        # L * D_i (where sum of D_is is the chans_out value returned by
        # get_output_shape_for)
        out_stacks = []
        for chosen_input, pools in index_spec:
            this_input = inputs[chosen_input]

            if all(len(p) == 1 for p in pools):
                # This is the fast path: we don't have to pool at all, because
                # there's precisely one element in each pool. This always
                # happens when constructing action modules, which never need
                # any sort of pooling.
                # FIXME: speed this up even further by doing combined gather &
                # concat, instead of running concat separately (actually,
                # before doing that, look at TF logs to see whether it's likely
                # to be faster!).
                np_indices = np.array([idx for idx, in pools], dtype=np.int32)
                assert np_indices.ndim == 1, np_indices.shape
                this_stack = tf.gather(this_input, np_indices, axis=1)
                out_stacks.append(this_stack)
            else:
                # Slow path. Need to make new array & reduce over the first
                # axis with segment_max.
                trans_input = tf.transpose(this_input, (1, 0, 2))
                flat_pools = sum(pools, [])
                pool_segments = sum(
                    ([r] * len(pools[r]) for r in range(len(pools))), [])
                gathered = tf.gather(trans_input, flat_pools, axis=0)
                if all(len(p) > 0 for p in pools):
                    # we can never end up with empty pools, so we just do
                    # segment_max, which seems to be a little faster
                    trans_pooled = tf.segment_max(gathered, pool_segments)
                    trans_pooled.set_shape((len(pools), trans_pooled.shape[1],
                                            trans_pooled.shape[2]))
                else:
                    # TODO: is there a faster way of doing this than
                    # unsorted_segment_max? I want something that will
                    # GRACEFULLY deal with missing segments; segment_max
                    # inserts 0 for missing interior segments (?!) and simply
                    # omits empty segments on the end (since it doesnt' know
                    # about them).
                    trans_pooled = tf.unsorted_segment_max(
                        gathered, pool_segments, len(pools))
                    # unsorted_segment_max will return FLOAT_MIN or something
                    # for empty segments; we get around that problem by
                    # clamping the value to the minimum possible activation
                    # output from the last layer
                    assert NONLINEARITY == 'elu', \
                        'minimum value of -1 is dependent on using elu'
                trans_pooled = tf.maximum(trans_pooled, -1.0)
                pooled = tf.transpose(trans_pooled, (1, 0, 2))
                out_stacks.append(pooled)

        # concat along input channels
        if extra_chans is not None:
            all_cat = out_stacks + list(extra_chans)
        else:
            all_cat = out_stacks
        rv = tf.concat(all_cat, axis=2)
        if debug:
            this_in_shape = tf.shape(this_input)
            in_bs = this_in_shape[0]
            check_bs = tf.assert_equal(tf.shape(rv)[0],
                                       in_bs,
                                       message="returned batch size doesn't "
                                       "match that of last input")
            check_rank = tf.assert_rank(rv, 3, message='return rank is not 3')
            with tf.control_dependencies([check_bs, check_rank]):
                rv = tf.identity(rv)
        return rv


class PolicyModel(metaclass=ABCMeta):
    """Base class for all policy models (MLP, ASNet, etc.). Very simple for
    now."""

    @property
    @abstractmethod
    def input_ph(self):
        """Should return a `tf.placeholder` representing input vector."""
        pass

    @property
    @abstractmethod
    def act_dist(self):
        """Should return a `tf.tensor` representing distribution over actions.
        Value depends on `self.input_ph`."""
        pass


class PropNetwork(PolicyModel):
    """Concrete implementation of a proposition/action network """

    @can_profile
    def __init__(self, weight_manager, problem_meta, dropout=0.0, debug=False):
        self._weight_manager = weight_manager
        self._prob_meta = problem_meta
        self._debug = debug
        # I tried ReLU, tanh, softplus, & leaky ReLU before settling on ELU for
        # best combination of numeric stability + sample efficiency
        self.nonlinearity = getattr(tf.nn, NONLINEARITY)
        # should we include skip connections?
        self.skip = self._weight_manager.skip

        self.dropout = dropout

        with tf.name_scope('asnet'):
            self._make_network()

    @property
    def act_dist(self):
        return self._act_dist

    @property
    def input_ph(self):
        return self._input_ph

    @can_profile
    def _make_network(self):
        hidden_sizes = self._weight_manager.hidden_sizes
        dom_meta = self._weight_manager.dom_meta
        prob_meta = self._prob_meta

        # input vector spec:
        #
        # |<--num_acts-->|<--k*num_acts-->|<--num_props-->|
        # | action mask  |  action data   | propositions  |
        #
        # 1) `action_mask` tells us whether actions are enabled
        # 2) `action_data` is passed straight to action modules
        # 3) `propositions` tells us what is and isn't true
        #
        # Reminder: this convoluted input shape is required solely because of
        # rllab's inflexible input conventions (it can only take a single
        # vector per state).
        #
        # FIXME: now that I'm not using RLLab any more, how about I change this
        # input convention around so that it actually makes sense? Should
        # collect inputs for each action module in Numpy vectors, then pass
        # them directly to the network as different k/v pairs in a feed_dict.

        mask_size = prob_meta.num_acts
        extra_data_dim = self._weight_manager.extra_dim
        extra_size = extra_data_dim * prob_meta.num_acts
        prop_size = prob_meta.num_props
        in_dim = mask_size + extra_size + prop_size
        self._input_ph = tf.placeholder(tf.float32, shape=(None, in_dim))

        def act_extra_inner(in_vec):
            act_vecs = in_vec[:, mask_size:mask_size + extra_size]
            out_shape = (-1, prob_meta.num_acts, extra_data_dim)
            return tf.reshape(act_vecs, out_shape)

        def obs_inner(in_vec):
            prop_truth = in_vec[:, mask_size + extra_size:, None]
            # FIXME: it doesn't make sense to mess with goal vectors here; that
            # should be ActionDataGenerator's job, or whatever. Should be
            # passed in as part of network input, not fixed as  TF constant!
            goal_vec = [
                float(prop in prob_meta.goal_props)
                for prop in prob_meta.bound_props_ordered
            ]

            assert sum(goal_vec) == len(prob_meta.goal_props)
            assert any(goal_vec), 'there are no goals?!'
            assert not all(goal_vec), 'there are no goals?!'

            # apparently this broadcasts (hooray!)
            tf_goals = tf.constant(goal_vec)[None, :, None]
            batch_size = tf.shape(prop_truth)[0]
            tf_goals_broad = tf.tile(tf_goals, (batch_size, 1, 1))
            l_obs = tf.concat([prop_truth, tf_goals_broad], axis=2)
            return l_obs

        l_obs = obs_inner(self.input_ph)

        pred_dict = self._split_input(l_obs)
        if extra_data_dim > 0:
            l_act_extra = act_extra_inner(self.input_ph)
            extra_dict = self._split_extra(l_act_extra)
        else:
            extra_dict = None

        # this is useful for getting values from ALL action/proposition layers
        self.act_layers = []
        self.prop_layers = []
        self.action_layer_input = {}

        # is this the input layer? (gets set to False on first loop)
        is_input = True

        # hidden layers
        prev_act_dict = {}
        prev_pred_dict = {}
        for hid_idx, hid_sizes in enumerate(hidden_sizes):
            act_size, prop_size = hid_sizes

            act_dict = {}
            for unbound_act in dom_meta.unbound_acts:
                act_dict[unbound_act] = self._make_action_module(
                    pred_dict,
                    unbound_act,
                    hid_idx,
                    dropout=self.dropout,
                    extra_dict=extra_dict,
                    prev_layer=prev_act_dict.get(unbound_act, None),
                    save_input=is_input)
            is_input = False
            self.act_layers.append(act_dict)
            prev_act_dict = act_dict

            pred_dict = {}
            for pred_name in dom_meta.pred_names:
                pred_dict[pred_name] = self._make_prop_module(
                    act_dict,
                    pred_name,
                    hid_idx,
                    prev_layer=prev_pred_dict.get(pred_name, None),
                    dropout=self.dropout,
                )
            self.prop_layers.append(pred_dict)
            prev_pred_dict = pred_dict

        # final (action) layer
        finals = {}
        for unbound_act in dom_meta.unbound_acts:
            finals[unbound_act] = self._make_action_module(
                pred_dict,
                unbound_act,
                len(hidden_sizes),
                nonlinearity=tf.identity,
                prev_layer=prev_act_dict.get(unbound_act, None),
                # can't have ANY dropout in final layer!
                dropout=0.0,
                extra_dict=extra_dict,
            )
        self.act_layers.append(finals)

        l_pre_softmax = self._merge_finals(finals)
        l_mask = self.input_ph[:, :mask_size]
        # voila!
        self._act_dist = masked_softmax(l_pre_softmax, l_mask)

    def _split_input(self, obs_layer):
        """Splits an observation layer up into appropriate proposition
        layers."""
        # TODO: comment below about _merge_finals ugliness also applies here
        prob_meta = self._prob_meta
        prop_to_flat_input_idx = {
            prop: idx
            for idx, prop in enumerate(prob_meta.bound_props_ordered)
        }
        rv = {}

        for pred_name in prob_meta.domain.pred_names:
            sub_props = prob_meta.pred_to_props(pred_name)
            gather_inds = []
            for sub_prop in sub_props:
                to_look_up = prop_to_flat_input_idx[sub_prop]
                gather_inds.append(to_look_up)

            rv[pred_name] = tf.gather(obs_layer,
                                      gather_inds,
                                      axis=1,
                                      name='split_input/' + pred_name)

        return rv

    def _split_extra(self, extra_data):
        """Sometimes we also have input data which goes straight to the
        network. We need to split this up into an unbound action->tensor
        dictionary just like the rest."""
        prob_meta = self._prob_meta
        out_dict = {}
        for unbound_act in prob_meta.domain.unbound_acts:
            ground_acts = prob_meta.schema_to_acts(unbound_act)
            sorted_acts = sorted(ground_acts,
                                 key=prob_meta.act_to_schema_subtensor_ind)
            if len(sorted_acts) == 0:
                # FIXME: make this message scarier
                print("no actions for schema %s?" % unbound_act.schema_name)
            # these are the indices which we must read and concatenate
            tensor_inds = [
                # TODO: make this linear-time (or linearithmic) by using a dict
                prob_meta.bound_acts_ordered.index(act) for act in sorted_acts
            ]

            out_dict[unbound_act] = tf.gather(extra_data,
                                              tensor_inds,
                                              axis=1,
                                              name='split_extra/' +
                                              unbound_act.schema_name)

        return out_dict

    @can_profile
    def _merge_finals(self, final_acts):
        prob_meta = self._prob_meta
        # we make a huge tensor of actions that we'll have to reorder
        sorted_final_acts = sorted(final_acts.items(), key=lambda t: t[0])
        # also get some metadata about which positions in tensor correspond to
        # which schemas
        unbound_to_super_ind = {
            t[0]: idx
            for idx, t in enumerate(sorted_final_acts)
        }
        # indiv_sizes[i] is the number of bound acts associated with the i-th
        # schema
        indiv_sizes = [
            len(prob_meta.schema_to_acts(ub)) for ub, _ in sorted_final_acts
        ]
        # cumul_sizes[i] is the sum of the number of ground actions associated
        # with each action schema *before* the i-th schema
        cumul_sizes = np.cumsum([0] + indiv_sizes)
        # this stores indices that we have to look up
        gather_list = []
        for ground_act in prob_meta.bound_acts_ordered:
            subact_ind = prob_meta.act_to_schema_subtensor_ind(ground_act)
            superact_ind = unbound_to_super_ind[ground_act.prototype]
            actual_ind = cumul_sizes[superact_ind] + subact_ind
            assert 0 <= actual_ind < prob_meta.num_acts, \
                "action index %d for %r out of range [0, %d)" \
                % (actual_ind, ground_act, prob_meta.num_acts)
            gather_list.append(actual_ind)

        # now let's actually build and reorder our huge tensor of action
        # selection probs
        cat_super_acts = tf.concat([t[1] for t in sorted_final_acts],
                                   axis=1,
                                   name='merge_finals/cat')
        rv = tf.gather(cat_super_acts[:, :, 0],
                       np.array(gather_list),
                       axis=1,
                       name='merge_finals/reorder')

        return rv

    def _finite_check(self, vec):
        """Create a TF assertion op to ensure that all elements of given vector
        are finite (i.e. not NaN or inf)."""
        is_finite = tf.is_finite(vec)
        # activations higher than 1B are insane
        is_small = tf.abs(vec) < 1e9
        all_finite = tf.reduce_all(is_finite)
        all_small = tf.reduce_all(is_small)
        frac_finite = tf.reduce_mean(tf.cast(is_finite, tf.float32))
        frac_small = tf.reduce_mean(tf.cast(is_small, tf.float32))
        cond = tf.reduce_all([all_finite, all_small])
        check_normed_op = tf.Assert(
            cond, [all_finite, frac_finite, all_small, frac_small],
            name='check_finite')
        with tf.control_dependencies([check_normed_op]):
            return tf.identity(vec)

    def _apply_conv_matmul(self, conv_input, W):
        reshaped = tf.reshape(conv_input, (-1, conv_input.shape[2]))
        conv_result_reshaped = tf.matmul(reshaped, W)
        conv_shape = tf.shape(conv_input)
        batch_size = conv_shape[0]
        # HACK: if conv_input.shape[1] is not Dimension(None) (i.e. if it's
        # known) then I want to keep that b/c it will help shape inference;
        # otherwise I want to use conv_shape[1], which will make the reshape
        # succeed. It turns out the best way I can see to compare
        # conv_input.shape[1] to Dimension(None) is to abuse comparison by
        # checking whether conv_input.shape[1] >= 0 returns None or True (!!).
        # This is stupid, but I can't see a better way of doing it.
        width = conv_shape[1] if (conv_input.shape[1] >= 0) is None \
            else conv_input.shape[1]
        out_shape = (batch_size, width, W.shape[1])
        conv_result = tf.reshape(conv_result_reshaped, out_shape)
        return conv_result

    @can_profile
    def _make_action_module(self,
                            prev_dict,
                            unbound_act,
                            layer_num,
                            nonlinearity=None,
                            dropout=0.0,
                            extra_dict=None,
                            *,
                            prev_layer=None,
                            save_input=False):
        # TODO: can I do all of this index-futzing just once, instead of each
        # time I need to make an action module? Same applies to proposition
        # modules. Will make construction much faster (not that it's very
        # expensive at the moment...).
        name_pfx = 'act_mod_%s_%d' % (unbound_act.schema_name, layer_num)
        prob_meta = self._prob_meta
        dom_meta = self._weight_manager.dom_meta

        # sort input layers so we can index into them properly
        pred_to_tensor_idx, prev_inputs = self._sort_inputs(prev_dict)

        # this tells us how many channels our input will need
        index_spec = []
        dom_rel_preds = dom_meta.rel_pred_names(unbound_act)
        for act_pred_idx, arg_pred in enumerate(dom_rel_preds):
            pools = []
            for ground_act in prob_meta.schema_to_acts(unbound_act):
                # we're looking at the act_pred_idx-th relevant proposition
                bound_prop = prob_meta.rel_props(ground_act)[act_pred_idx]
                prop_idx = prob_meta.prop_to_pred_subtensor_ind(bound_prop)
                # we're only "pooling" over one element (the proposition
                # features)
                pools.append([prop_idx])

            # which tensor do we need to pick this out of?
            tensor_idx = pred_to_tensor_idx[arg_pred]
            index_spec.append((tensor_idx, pools))

        extra_chans = []
        if layer_num == 0 and extra_dict is not None:
            # first action layer, so add in extra data
            act_data = extra_dict[unbound_act]
            extra_chans.append(act_data)
        if layer_num > 0 and self.skip:
            assert prev_layer is not None, \
                "act mod in L%d not supplied previous acts for skip conn" \
                % layer_num
            extra_chans.append(prev_layer)
        elif layer_num == 0 and self.skip:
            assert prev_layer is None, "ugh this shouldn't happen in layer 0"
        if USE_CUSTOM_MULTI_GATHER_CONCAT:
            with tf.name_scope(name_pfx + '/mgc'):
                mgc_inputs = []
                mgc_elem_indices = []
                for tensor_idx, pools in index_spec:
                    mgc_inputs.append(prev_inputs[tensor_idx])
                    elem_inds = [p for p, in pools]
                    mgc_elem_indices.append(
                        tf.constant(elem_inds, dtype=tf.int64))
                for extra_chan in extra_chans:
                    mgc_inputs.append(extra_chan)
                    extra_chan_width = tf.cast(
                        tf.shape(extra_chan)[1], tf.int64)
                    mgc_elem_indices.append(
                        tf.range(extra_chan_width, dtype=tf.int64))
                    # helps out shape inference if extra_chan.shape[1] is known
                    mgc_elem_indices[-1].set_shape(extra_chan.shape[1])
                conv_input = multi_gather_concat(mgc_inputs, mgc_elem_indices)
        else:
            conv_input = pick_pool_and_stack(prev_inputs,
                                             index_spec,
                                             name=name_pfx + '/cat',
                                             debug=self._debug,
                                             extra_chans=extra_chans)
        if save_input:
            # hack so that I can get at input to the convolution when
            # displaying layer-by-layer network activations (it's easier to do
            # it this way)
            self.action_layer_input[unbound_act] = conv_input
        W, b = self._weight_manager.act_weights[layer_num][unbound_act]
        if nonlinearity is None:
            nonlinearity = self.nonlinearity
        with tf.name_scope(name_pfx + '/conv'):
            conv_result = self._apply_conv_matmul(conv_input, W)
            rv = nonlinearity(conv_result + b[None, :])
            # FIXME: is this really necessary after every conv module?
            if self._debug:
                rv = self._finite_check(rv)

        if dropout > 0:
            rv = tf.nn.dropout(rv, rate=dropout, name=name_pfx + '/drop')

        return rv

    @can_profile
    def _make_prop_module(
            self,
            prev_dict,
            pred_name,
            layer_num,
            *,
            prev_layer=None,
            dropout=0.0,
    ):
        # TODO: comment about indexing futzing also applies here---should try
        # to do it just once.
        name_pfx = 'prop_mod_%s_%d' % (pred_name, layer_num)
        prob_meta = self._prob_meta
        dom_meta = self._weight_manager.dom_meta
        act_to_tensor_idx, prev_inputs = self._sort_inputs(prev_dict)
        index_spec = []
        pred_rela_slots = dom_meta.rel_act_slots(pred_name)
        for rel_unbound_act_slot in pred_rela_slots:
            pools = []
            for prop in prob_meta.pred_to_props(pred_name):
                # we're looking at the act_pred_idx-th relevant proposition
                rel_slots = prob_meta.rel_act_slots(prop)
                ground_acts = [
                    ground_act for unbound_act, slot, ground_acts in rel_slots
                    for ground_act in ground_acts
                    if (unbound_act, slot) == rel_unbound_act_slot
                ]
                act_inds = [
                    prob_meta.act_to_schema_subtensor_ind(ground_act)
                    for ground_act in ground_acts
                ]
                pools.append(act_inds)

            tensor_idx = act_to_tensor_idx[rel_unbound_act_slot[0]]
            index_spec.append((tensor_idx, pools))

        extra_chans = []
        if layer_num > 0 and self.skip:
            assert prev_layer is not None, \
                "pred mod in L%d not supplied previous acts for skip conn" \
                % layer_num
            extra_chans.append(prev_layer)
        elif layer_num == 0 and self.skip:
            assert prev_layer is None, "ugh this shouldn't happen in layer 0"
        if USE_CUSTOM_MULTI_POOL_CONCAT:
            # use a custom fused op to create input to prop module
            with tf.name_scope(name_pfx + '/mpc'):
                mpc_inputs = []
                mpc_ragged_pools = []
                for tensor_idx, py_pools in index_spec:
                    mpc_inputs.append(prev_inputs[tensor_idx])
                    flat_pools = sum(py_pools, [])
                    pool_lens = [len(p) for p in py_pools]
                    ragged_pool = tf.cast(
                        tf.RaggedTensor.from_row_lengths(
                            flat_pools, pool_lens), tf.int64)
                    mpc_ragged_pools.append(ragged_pool)
                assert NONLINEARITY == 'elu', \
                    'minimum value of -1 is dependent on using elu'
                min_value = -1.0
                conv_input = multi_pool_concat(mpc_inputs, mpc_ragged_pools,
                                               min_value)
                if extra_chans:
                    # TODO: also test adding this directly to
                    # multi_pool_concat; is it any slower?
                    conv_input = tf.concat([conv_input, *extra_chans], axis=2)
        else:
            conv_input = pick_pool_and_stack(prev_inputs,
                                             index_spec,
                                             extra_chans=extra_chans,
                                             name=name_pfx + '/cat')
        W, b = self._weight_manager.prop_weights[layer_num][pred_name]
        with tf.name_scope(name_pfx + '/conv'):
            conv_result = self._apply_conv_matmul(conv_input, W)
            rv = self.nonlinearity(conv_result + b[None, :])
            if self._debug:
                rv = self._finite_check(rv)

        if dropout > 0:
            rv = tf.nn.dropout(rv, rate=dropout, name=name_pfx + '/drop')

        return rv

    def _sort_inputs(self, prev_dict):
        # Sort order is kind of arbitrary. Main thing is that order implied by
        # pred_to_tensor_idx must exactly match that of prev_inputs, as
        # pred_to_tensor_idx is used to index into prev_inputs.
        input_items_sorted = sorted(prev_dict.items(), key=lambda p: p[0])
        pred_to_tensor_idx = {
            p[0]: idx
            for idx, p in enumerate(input_items_sorted)
        }
        prev_inputs = [tensor for _, tensor in input_items_sorted]
        return pred_to_tensor_idx, prev_inputs
