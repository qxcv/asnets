#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;

///////////////////////////////////////////////////////////////////////////////
// OP REGISTRATION
///////////////////////////////////////////////////////////////////////////////

REGISTER_OP("MultiGatherConcat")
    // ith tensor is batch_size * num_props[i] * num_channels[i]
    .Input("inputs: N * float")
    // each elem_indices is of size num_acts & corresponds to an element of
    // inputs
    .Input("elem_indices: N * int64")
    // batch_size * num_acts * sum(channel count of each inputs in inputs)
    .Output("output: float")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;
      auto N = c->num_inputs() / 2;
      CHECK_EQ(c->num_inputs(), 2 * N);
      CHECK_GE(N, 1);

      // This function does minimal shape checks because ShapeHandle and
      // InferenceContext are really annoying to use. I'll do more checks in
      // Compute(), and more checks on the Python side. I suspect the *reason*
      // why checks are hard to perform is because it's ideal if ops can work
      // with inputs of arbitrary dimension (and perhaps even rank) at execution
      // time, and having heavy checks during shape inference could (annoyingly)
      // force people to specify types ahead of time.

      // get inputs shape
      DimensionHandle out_chan_dim = c->MakeDim(0);
      DimensionHandle out_batch_dim;
      for (int i = 0; i < N; ++i) {
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input_shape));
        TF_RETURN_IF_ERROR(
            c->Add(c->Dim(input_shape, 2), out_chan_dim, &out_chan_dim));
        out_batch_dim = c->Dim(input_shape, 0);
      }

      // now get remaining two dims of output shape; we iterate over all
      // elem_indices items to make sure they all have correct rank of 1
      ShapeHandle elem_inds_shape;
      for (int j = N; j < 2 * N; ++j) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(j), 1, &elem_inds_shape));
      }
      c->set_output(0, c->MakeShape({out_batch_dim, c->Dim(elem_inds_shape, 0),
                                     out_chan_dim}));
      return Status::OK();
    });

// no shape inference for this one
REGISTER_OP("MultiGatherConcatGrad")
    .Input("grad: float")
    // FIXME: orig_inputs are only needed for shape computation; is there a
    // faster way of doing this? (same comment is in asnet_ops.py)
    .Input("orig_inputs: N * float")
    .Input("elem_indices: N * int64")
    .Output("input_grads: N * float")
    .Attr("N: int >= 1");

REGISTER_OP("MultiPoolConcat")
    // ith tensor is again batch_size * num_acts[i] * num_channels[i]
    .Input("inputs: N * float")
    // encodes list of N ragged tensors in which ith ragged tensor is of size
    // num_props*?, and the jth row of that tensor indexes into elements of
    // inputs[i] that will be max pooled together to form the jth group of
    // output channels.
    .Input("elem_indices_values: N * int64")
    .Input("elem_indices_row_splits: N * int64")
    // used as base for the max pooling recursion; will never return a value
    // lower than this, and will also use this value as default output when a
    // row is empty
    .Input("min_value: float")
    // batch_size * num_props * sum(channel count of each inputs in inputs),
    // just like MultiGatherConcat
    .Output("output: float")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;
      auto N = (c->num_inputs() - 1) / 3;
      CHECK_EQ(c->num_inputs(), 3 * N + 1);
      CHECK_GE(N, 1);

      // input shape; this is duplicated from previous shape inference fn, &
      // doesn't need any special ragged tensor handling
      // FIXME: de-duplicate this once I figure out how to handle the stupid
      // TF_RETURN_IF_ERROR macro.
      DimensionHandle out_chan_dim = c->MakeDim(0);
      DimensionHandle out_batch_dim;
      for (int i = 0; i < N; ++i) {
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 3, &input_shape));
        TF_RETURN_IF_ERROR(
            c->Add(c->Dim(input_shape, 2), out_chan_dim, &out_chan_dim));
        out_batch_dim = c->Dim(input_shape, 0);
      }

      // iterate over the index value tensors to make sure they're all of rank 1
      ShapeHandle unused;
      for (int j = N; j < 2 * N; ++j) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(j), 1, &unused));
      }

      // iterate over the row splits to make sure they're all of rank 1
      ShapeHandle inds_split_shape;
      for (int k = 2 * N; k < 3 * N; ++k) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(k), 1, &inds_split_shape));
      }

      // now get output width by subtracting 1 from the size of the row splits
      DimensionHandle inds_width = c->Dim(inds_split_shape, 0);
      DimensionHandle const_1 = c->MakeDim(1);
      TF_RETURN_IF_ERROR(c->Max(inds_width, const_1, &inds_width));
      DimensionHandle out_width;
      TF_RETURN_IF_ERROR(c->Subtract(inds_width, const_1, &out_width));

      // finally, check that min_value is a scalar
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3 * N), 0, &unused));

      c->set_output(0, c->MakeShape({out_batch_dim, out_width, out_chan_dim}));

      return Status::OK();
    });

REGISTER_OP("MultiPoolConcatGrad")
    // grad of main loss w.r.t the one output of MultiPoolConcat
    .Input("grad: float")
    // these are the original inputs; we actually need all of them to compute
    // the gradient!
    .Input("inputs: N * float")
    .Input("elem_indices_values: N * int64")
    .Input("elem_indices_row_splits: N * int64")
    // we also need original output tensor to figure out which elements actually
    // contributed to the original result
    .Input("orig_output: float")
    // tensors giving gradient of main loss w.r.t each of the original inputs to
    // MultiPoolConcat
    .Output("input_grads: N * float")
    .Attr("N: int >= 1");

///////////////////////////////////////////////////////////////////////////////
// KERNELS (OP IMPLEMENTATIONS)
///////////////////////////////////////////////////////////////////////////////

// checking that # of out-of-bounds failures is 0
#define ASSERT_NO_OOB_FAILURES(failures)                                       \
  OP_REQUIRES(ctx, failures == 0,                                              \
              errors::InvalidArgument("op encountered ", failures,             \
                                      " out-of-bounds access(es)"))

// Makes a stupid 1D, single-element Eigen array for use as the first or second
// argument of tensor.slice(bound, extent). It's slightly annoying that Eigen
// wants such an array instead of auto-promoting integer types.
#define SLICE_ARG(scalar_value) (Eigen::array<int64, 1>{scalar_value})

// FIXME: how can macro below be converted to inline fn, or otherwise turned
// into something that's not just a macro? I'm mostly worried about
// OP_REQUIRES_OK, which won't work properly if I put it in an inline function
// (not the end of the world b/c I can define a new macro, but still annoying
// having to do that at all)

// common input processing for both MultiGatherConcatOp and
// MultiGatherConcatGradOp. Supports custom C++ variable name for inputs_list,
// and also lets you specify string that should take place of inputs_name when
// enumerating inputs to the op.
#define MULTI_GATHER_COMMON_INPUT_PROC(inputs_list, inputs_name)               \
  OpInputList inputs_list;                                                     \
  OP_REQUIRES_OK(ctx, ctx->input_list(inputs_name, &inputs_list));             \
  int N = inputs_list.size();                                                  \
  OP_REQUIRES(                                                                 \
      ctx, N >= 1,                                                             \
      errors::InvalidArgument("must have at least one input, got ", N));       \
  OpInputList elem_indices_list;                                               \
  OP_REQUIRES_OK(ctx, ctx->input_list("elem_indices", &elem_indices_list));    \
  OP_REQUIRES(                                                                 \
      ctx, N == elem_indices_list.size(),                                      \
      errors::InvalidArgument("number of 'inputs' (", N,                       \
                              ") must match number of 'elem_indices' (",       \
                              elem_indices_list.size(), ")"));                 \
  TensorShape elem_inds_shape;                                                 \
  elem_inds_shape = elem_indices_list[0].shape();                              \
  int64 out_width = elem_inds_shape.dim_size(0);                               \
  for (const auto &elem_indices : elem_indices_list) {                         \
    const auto &this_shape = elem_indices.shape();                             \
    OP_REQUIRES(ctx, elem_inds_shape == this_shape,                            \
                errors::InvalidArgument(                                       \
                    "all elem_indices must have same shape, but got shape ",   \
                    this_shape, " that doesn't match first shape ",            \
                    elem_inds_shape));                                         \
  }                                                                            \
  int64 batch_size = inputs_list[0].shape().dim_size(0);                       \
  std::vector<int64> out_slot_start_chan;                                      \
  int64 chan_sum = 0;                                                          \
  for (const auto &input_tensor : inputs_list) {                               \
    const auto &in_shape = input_tensor.shape();                               \
    OP_REQUIRES(ctx, in_shape.dim_size(0) == batch_size,                       \
                errors::InvalidArgument(                                       \
                    "expected uniform batch size ", batch_size,                \
                    "but got a batch of size ", in_shape.dim_size(0)));        \
    out_slot_start_chan.push_back(chan_sum);                                   \
    chan_sum += in_shape.dim_size(2);                                          \
  }

class MultiGatherConcatOp : public OpKernel {
public:
  explicit MultiGatherConcatOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    MULTI_GATHER_COMMON_INPUT_PROC(inputs_list, "inputs");

    // build output shape
    TensorShape out_shape;
    out_shape.AddDim(batch_size);
    out_shape.AddDim(out_width);
    out_shape.AddDim(chan_sum);

    // allocate output tensor, and also get a 2D view on it for easy slicing
    Tensor *out_tensor_tf;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor_tf));
    auto out_tensor = out_tensor_tf->tensor<float, 3>();
    out_tensor.setZero();

    // now process everything (mnemonics: "b" for batch, "c" for (output) cell,
    // "n" for one for the N sub-inputs)

    // FIXME: I'm keeping a fail count here & erroring out late to give compiler
    // some leeway to optimise; does it actually matter, though? If not, it
    // would smarter to error out here.
    int failures = 0;
    for (int64 b = 0; b < batch_size; ++b) {
      for (int64 c = 0; c < out_width; ++c) {
        for (int n = 0; n < N; ++n) {
          const auto elem_inds_tens = elem_indices_list[n].tensor<int64, 1>();
          const auto input_tens = inputs_list[n].tensor<float, 3>();
          int64 selected_input = elem_inds_tens(c);
          // safely set output slice using selected input slice
          if (selected_input < 0 || selected_input >= input_tens.dimension(1)) {
            ++failures;
          } else {
            auto out_start_chan = SLICE_ARG(out_slot_start_chan[n]);
            auto in_chans = SLICE_ARG(input_tens.dimension(2));
            const auto in_slice = input_tens.chip(b, 0).chip(selected_input, 0);
            ((out_tensor.template chip<0>(b)).template chip<0>(c))
                .slice(out_start_chan, in_chans) = in_slice;
          }
        }
      }
    }

    ASSERT_NO_OOB_FAILURES(failures);

    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiGatherConcat").Device(DEVICE_CPU),
                        MultiGatherConcatOp)

class MultiGatherConcatGradOp : public OpKernel {
public:
  explicit MultiGatherConcatGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    MULTI_GATHER_COMMON_INPUT_PROC(orig_inputs_list, "orig_inputs");

    const Tensor *grad_wrt_orig_output_tf;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_wrt_orig_output_tf));
    const auto grad_tensor = grad_wrt_orig_output_tf->tensor<float, 3>();
    const auto &gt_shape = grad_wrt_orig_output_tf->shape();
    TensorShape expected_grad_shape;
    expected_grad_shape.AddDim(batch_size);
    expected_grad_shape.AddDim(out_width);
    expected_grad_shape.AddDim(chan_sum);
    OP_REQUIRES(ctx, gt_shape == expected_grad_shape,
                errors::InvalidArgument("provided grad shape ", gt_shape,
                                        " does not match expected shape ",
                                        expected_grad_shape));

    // for tracking out-of-bounds accesses again
    int failures = 0;

    for (int n = 0; n < N; ++n) {
      // allocate output tensor for gradient w.r.t this original input
      const auto &orig_input_shape = orig_inputs_list[n].shape();
      Tensor *orig_input_grad_tf;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(n, orig_input_shape, &orig_input_grad_tf));
      auto orig_input_grad_tens = orig_input_grad_tf->tensor<float, 3>();
      // As far as I can tell, zero init does not happen automatically. We need
      // to do it manually so that gradient accumulation works properly.
      orig_input_grad_tens.setZero();

      // other things we'll need in this grad impl
      const auto elem_inds_tens = elem_indices_list[n].tensor<int64, 1>();
      auto out_start_chan = SLICE_ARG(out_slot_start_chan[n]);

      for (int64 b = 0; b < batch_size; ++b) {
        for (int64 c = 0; c < out_width; ++c) {
          int64 selected_input = elem_inds_tens(c);
          if (selected_input < 0 ||
              selected_input >= orig_input_grad_tens.dimension(1)) {
            ++failures;
          } else {
            auto in_chans = SLICE_ARG(orig_input_grad_tens.dimension(2));
            auto grad_slice =
                ((grad_tensor.template chip<0>(b)).template chip<0>(c))
                    .slice(out_start_chan, in_chans);
            // Accumulate back to input. This mirrors what we did on the forward
            // pass.
            orig_input_grad_tens.chip(b, 0).chip(selected_input, 0) +=
                grad_slice;
          }
        }
      }
    }

    ASSERT_NO_OOB_FAILURES(failures);

    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiGatherConcatGrad").Device(DEVICE_CPU),
                        MultiGatherConcatGradOp)

#define MULTI_POOL_COMMON_INPUT_PROC(ctx)                                      \
  /* grab inputs & compute channel sum, batch size, etc. */                    \
  OpInputList inputs_list;                                                     \
  OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs_list));                \
  int N = inputs_list.size();                                                  \
  int64 batch_size = inputs_list[0].shape().dim_size(0);                       \
  std::vector<int64> out_slot_start_chan;                                      \
  int64 chan_sum = 0;                                                          \
  for (const auto &input_tensor : inputs_list) {                               \
    const auto &in_shape = input_tensor.shape();                               \
    OP_REQUIRES(ctx, in_shape.dim_size(0) == batch_size,                       \
                errors::InvalidArgument(                                       \
                    "expected uniform batch size ", batch_size,                \
                    "but got a batch of size ", in_shape.dim_size(0)));        \
    out_slot_start_chan.push_back(chan_sum);                                   \
    chan_sum += in_shape.dim_size(2);                                          \
  }                                                                            \
  /* graph elem_indices_values & check that number of inputs in that */        \
  /* list makes sense (no need to check for 1D shape of each elem, since */    \
  /* that should have happened during shape inference) */                      \
  OpInputList elem_inds_values_list;                                           \
  OP_REQUIRES_OK(                                                              \
      ctx, ctx->input_list("elem_indices_values", &elem_inds_values_list));    \
  OP_REQUIRES(                                                                 \
      ctx, elem_inds_values_list.size() == N,                                  \
      errors::InvalidArgument("'inputs' length ", N,                           \
                              " does not match 'elem_indices_values' length ", \
                              elem_inds_values_list.size()));                  \
  /* do same for elem_inds_splits_list, w/ additional check to make sure */    \
  /* that split vector lengths match */                                        \
  OpInputList elem_inds_splits_list;                                           \
  OP_REQUIRES_OK(ctx, ctx->input_list("elem_indices_row_splits",               \
                                      &elem_inds_splits_list));                \
  OP_REQUIRES(ctx, elem_inds_splits_list.size() == N,                          \
              errors::InvalidArgument(                                         \
                  "'inputs' length ", N,                                       \
                  " does not match 'elem_indices_row_splits' length ",         \
                  elem_inds_splits_list.size()));                              \
  int64 in_width = elem_inds_splits_list[0].shape().dim_size(0);               \
  int64 out_width = std::max(in_width, 1ll) - 1ll;                             \
  for (const auto &inds_split : elem_inds_splits_list) {                       \
    const auto &split_shape = inds_split.shape();                              \
    OP_REQUIRES(                                                               \
        ctx, split_shape.dim_size(0) == out_width + 1,                         \
        errors::InvalidArgument("expected all index tensors to have length ",  \
                                out_width + 1, ", but got one with length ",   \
                                split_shape.dim_size(0)));                     \
  }

class MultiPoolConcatOp : public OpKernel {
public:
  explicit MultiPoolConcatOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    MULTI_POOL_COMMON_INPUT_PROC(ctx);

    // grab min_value placeholder (we know this is a scalar/rank 0 tensor)
    const Tensor *min_value;
    OP_REQUIRES_OK(ctx, ctx->input("min_value", &min_value));
    auto mv_scalar = min_value->scalar<float>();

    // allocate output tensor & fill it with min_value
    TensorShape out_shape;
    out_shape.AddDim(batch_size);
    out_shape.AddDim(out_width);
    out_shape.AddDim(chan_sum);
    Tensor *out_tensor_tf;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out_tensor_tf));
    auto out_tens = out_tensor_tf->tensor<float, 3>();
    // fill with min_value so we never return anything less than that
    // HACK: will this work on GPU?
    out_tens.setConstant(*mv_scalar.data());

    int failures = 0;
    for (int64 b = 0; b < batch_size; ++b) {
      for (int64 c = 0; c < out_width; ++c) {
        for (int n = 0; n < N; ++n) {
          const auto inds_split_tens =
              elem_inds_splits_list[n].tensor<int64, 1>();
          const auto inds_value_tens =
              elem_inds_values_list[n].tensor<int64, 1>();
          const auto input_tens = inputs_list[n].tensor<float, 3>();
          const int64 in_width = input_tens.dimension(1);
          const auto in_chans = SLICE_ARG(input_tens.dimension(2));
          const auto out_start_chan = SLICE_ARG(out_slot_start_chan[n]);
          int64 vals_start = inds_split_tens(c),
                vals_end = inds_split_tens(c + 1);
          for (int64 v = vals_start; v < vals_end; ++v) {
            const auto selected_input = inds_value_tens(v);
            if (selected_input < 0 || selected_input >= in_width) {
              ++failures;
            } else {
              auto out_slice =
                  ((out_tens.template chip<0>(b)).template chip<0>(c))
                      .slice(out_start_chan, in_chans);
              const auto in_slice =
                  input_tens.chip(b, 0).chip(selected_input, 0);
              out_slice = out_slice.cwiseMax(in_slice);
            }
          }
        }
      }
    }

    ASSERT_NO_OOB_FAILURES(failures);

    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiPoolConcat").Device(DEVICE_CPU),
                        MultiPoolConcatOp)

class MultiPoolConcatGradOp : public OpKernel {
public:
  explicit MultiPoolConcatGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    // common input processing code
    MULTI_POOL_COMMON_INPUT_PROC(ctx);

    const Tensor *grad, *orig_output;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad));
    OP_REQUIRES_OK(ctx, ctx->input("orig_output", &orig_output));
    // check that shapes of grad and orig_output match each other, and also
    // match what we expect output shape should be
    OP_REQUIRES(ctx, grad->shape() == orig_output->shape(),
                errors::InvalidArgument("grad shape ", grad->shape(),
                                        " does not match orig_output shape ",
                                        orig_output->shape()));
    TensorShape expected_orig_out_shape;
    expected_orig_out_shape.AddDim(batch_size);
    expected_orig_out_shape.AddDim(out_width);
    expected_orig_out_shape.AddDim(chan_sum);
    OP_REQUIRES(ctx, orig_output->shape() == expected_orig_out_shape,
                errors::InvalidArgument(
                    "orig_output shape ", orig_output->shape(),
                    "does not match expected shape ", expected_orig_out_shape));
    // convert to tensor
    const auto grad_tens = grad->tensor<float, 3>();
    const auto orig_output_tens = orig_output->tensor<float, 3>();

    // Need to divide gradient *evenly* among selected elements, instead of
    // choosing some other subgradient (e.g BP-ing full gradient to all of
    // them). Using a different subgradient may not allow you to deal with
    // duplicate inputs properly! See _UnsortedSegmentMaxGrad in TF for example
    // of what you SHOULD do.

    for (int n = 0; n < N; ++n) {
      // allocate output gradient accumulator & set it to zero
      Tensor *orig_input_grad_tf;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(n, inputs_list[n].shape(),
                                               &orig_input_grad_tf));
      auto orig_input_grad_tens = orig_input_grad_tf->tensor<float, 3>();
      orig_input_grad_tens.setZero();

      // other values we'll need
      const auto inds_split_tens = elem_inds_splits_list[n].tensor<int64, 1>();
      const auto inds_value_tens = elem_inds_values_list[n].tensor<int64, 1>();
      const auto input_tens = inputs_list[n].tensor<float, 3>();
      const int64 in_width = input_tens.dimension(1);
      const auto in_chans = SLICE_ARG(input_tens.dimension(2));
      const auto out_start_chan = SLICE_ARG(out_slot_start_chan[n]);

      // a count vector to see how many inputs achieve the maximum in each
      // channel
      Eigen::Tensor<int64, 1, Eigen::RowMajor> achiever_count(
          input_tens.dimension(2));

      for (int64 b = 0; b < batch_size; ++b) {
        for (int64 c = 0; c < out_width; ++c) {
          // reset achiever count
          achiever_count.setZero();
          int64 vals_start = inds_split_tens(c),
                vals_end = inds_split_tens(c + 1);
          // we use .eval() to make sure these are actually evaluated
          const auto actual_output =
              ((orig_output_tens.template chip<0>(b)).template chip<0>(c))
                  .slice(out_start_chan, in_chans)
                  .eval();
          for (int64 v = vals_start; v < vals_end; ++v) {
            const int64 selected_input = inds_value_tens(v);
            // this mode of error handling will be harder to port to CUDA than
            // `failures++` thing, but using it anyway here b/c it's easier :)
            OP_REQUIRES(ctx, selected_input >= 0 && selected_input < in_width,
                        errors::InvalidArgument(
                            "encountered out-of-bound index ", selected_input));
            // check whether this matches output in each dim
            auto this_input = (input_tens.template chip<0>(b))
                                  .template chip<0>(selected_input);
            // TODO: should I convert this comparison & the one below to use (>=
            // max - eps) instead of (== max)? May be good defensive
            // programming.
            achiever_count += (this_input == actual_output).cast<int64>();
          }

          // now scale grad slice by the achiever_count (bounded down by 1 to
          // prevent overflow) & accumulate as necessary
          const auto grad_slice =
              ((grad_tens.template chip<0>(b)).template chip<0>(c))
                  .slice(out_start_chan, in_chans);
          const auto grad_slice_scaled =
              (grad_slice / achiever_count.cwiseMax(1ll).cast<float>()).eval();
          for (int64 v = vals_start; v < vals_end; ++v) {
            const int64 selected_input = inds_value_tens(v);
            auto this_input = (input_tens.template chip<0>(b))
                                  .template chip<0>(selected_input);
            auto max_mask = (this_input == actual_output).cast<float>();
            auto out_grad_slice = (orig_input_grad_tens.template chip<0>(b))
                                      .template chip<0>(selected_input);
            out_grad_slice += max_mask * grad_slice_scaled;
          }
        }
      }
    }

    return;
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiPoolConcatGrad").Device(DEVICE_CPU),
                        MultiPoolConcatGradOp)
