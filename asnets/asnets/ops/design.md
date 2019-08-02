# Design doc for custom pick-stack/pick-pool-stack ops

## The pick-stack op

*Warning:* I'm sceptical of the utility of this op; I think it would be easier
to do a single `concat` on all proposition representations, then do a single
`gather_nd` for each action schema to build the input for the action schema.
That would result in a factor of `num_action_schemas` fewer concat ops, and
`average_input_preds_per_schema` fewer gather ops (since each action schema
should only need one `gather_nd`). The main downsides to this are (1) I'm not
100% sure that `gather_nd` will do what I want it to, and (2) this won't let me
deal with channel counts that vary for each proposition output (which I may want
to deal with in the future). I do, however, think that creating a pick-stack op
could be a useful educational exercise, since it will force me to think about
what the `pick-pool-stack` op should do.

This is the fast path of the `pick_pool_and_stack` function in `models.py`,
where it just selects some vectors from a bunch of inputs & then concatenates
them together into a single output vector.

**Inputs:**

- `inputs`: `N` different `float` inputs to pick from, each of which will be a
  `B*?*C` tensor, where `B` is batch size, `C` is channel count, and the middle
  `?` is a proposition count for the corresponding predicate (which will vary
  for each of the `N` inputs).
- `elem_indices`: `N` different `int32` input vectors. The `i`th vector of
  `elem_indices` is `A`-dimensional (where `A` is the number of actions in the
  output action schema), and contains indices into the `inputs_indices[j]`th
  tensor in `inputs`. Those will all be collected & concatenated into a single
  output.

**Outputs:**

- `outputs`: a single `float` output that is `B*A*(J*C)`, where `A` corresponds
  to the number of actions we have, and where `J` is of course the number of
  prop representations that get joined together to create this act
  representation.

For maximum op-fusing, maybe I can even add a `prev_layer** tensor that gets
directly concatenated to the output? That would allow me to implement skip
connections without the need for a separate concatenation after the
`pick_pool_and_stack** layer.

## Gradient of the pick-stack op

Gradient here is very simple since we're just concatenating.

**Next-layer grads that we use to produce prev-layer grads:**

- `dE/d(outputs)`: single tensor of same size as `outputs` representing gradient
  of error with respect to each of those outputs.

**Pick-stack inputs that we produce gradients w.r.t:**

- `dE/d(inputs)`: will produce `N` float tensors, with exactly same shapes as
  `inputs` had on the forward pass. The `n`th grad tensor will be `B*?*C*`, and
  the `C`-dimensional final-axis vector in `dE/d(inputs)[n][b][i]` will just be
  a sum of the `dE/d(outputs)[m][b][j]` grads that depended on it in the forward
  pass (for each relevant `m` and `j` given by `input_indices` and
  `elem_indices`).
- `dE/d(input_indices)`: not defined.
- `dE/d(elem_indices)`: not defined.

## The pick-pool-stack op

This is the slow(er) path of the `pick_pool_and_stack` function in `models.py`,
where the op also needs to max-pool across a bunch of different inputs (in fact,
a variable number of inputs per output!) and then concatenate. Probably this
also offers greater speedup opportunities, since I'm totally avoiding having to
explicitly instantiate a huge tensor of outputs. I suspect I'll need to use
ragged input tensors to store the pool groups.

**Inputs:**

- `inputs`: again, `N` different `float` tensors to pick from, each `B*?*C` &
  each produced by the previous-layer action modules.
- `inputs_indices`: a `J`-dimensional vector containing indices (in `[0, N)`)
  into `inputs`, with analogous semantics to `inputs_indices` in the pick-stack
  op.
- `pools`: `J` different ragged tensors corresponding to channel groups for
  outputs. The `J`th ragged tensor

**FIXME:** the ragged tensor part of the above spec isn't quite correct, since
"ragged tensors" are a Python-side abstraction. To actually support ragged
tensors, you need a Python-side wrapper that extracts values & row splits, and a
C++-side op that takes raw values & row splits and returns new values & row
splits (if your output is also ragged, that is). See the [`RaggedGather`
op](https://github.com/tensorflow/tensorflow/blob/c45be9283490d8e8d7b67074bfbdc9f24d67bc02/tensorflow/core/ops/ragged_array_ops.cc#L31-L42)
(C++) and the [ragged `gather()`
wrapper](https://github.com/tensorflow/tensorflow/blob/f7415d1efbe544c03107141eb3cc73714c5276bf/tensorflow/python/ops/ragged/ragged_gather_ops.py#L37-L121)
(Python). That should still not be a huge problem, though.

Aside: I remain confused as to why TF's `UnsortedSegmentMax` appeared to be so
much slower than `SegmentMax` when I last tested it. The [TF implementation of
`SegmentMax`](https://github.com/tensorflow/tensorflow/blob/4e1b2f4d8f97910c2f985759417d893b64748972/tensorflow/core/kernels/segment_reduction_ops.cc#L318-L319)
just calls out to [Eigen's
`MaxReducer`](https://eigen.tuxfamily.org/dox/unsupported/TensorFunctors_8h_source.html),
which isn't obviously magic (e.g no explicit use of AVX etc.).
`UnsortedSegmentMax` is implemented in a [roughly equivalent
way](https://github.com/tensorflow/tensorflow/blob/4e1b2f4d8f97910c2f985759417d893b64748972/tensorflow/core/kernels/segment_reduction_ops.cc#L520-L522),
but by specialising on TensorFlow's `UnsortedSegmentReductionOp` instead of
Eigen's machinery for reduction ops. Maybe when Eigen & TF are compiled, all the
inlining bottoms out to something more amenable to optimisation in Eigen's case
than in TF's case? Definitely worth sticking the core of my own implementation
into Godbolt just to check, either way.

## Gradient of the pick-pool-stack op

Should be same as pick-stack op, but only backpropagating to elements that
attained the recovered max value.

Aside: I'm pretty sure it won't be any faster to output argmax indices, b/c I'll
need to store a huge (and perhaps even variable-size) volume of them, which may
be so slow to allocate & manage that it eats up gains from not looking at the
inputs again. Also, looking at the input will make the backwards pass only
slightly slower than forward pass (b/c it has an extra tensor to read into cache
in order to check values), which is basically fine by me b/c memory overhead is
currently much more important than backprop speed.

## Some notes on the `Gather` op

The `Gather` op is a reasonable prototype for the ops that I'm about to write,
since it does roughly the same thing modulo pooling. The actual TF `Gather` op
mostly just calls out to
[`GatherFunctor`](https://github.com/tensorflow/tensorflow/blob/4e15557ac4c92f219f51ed94883d1e47dca88417/tensorflow/core/kernels/gather_functor.h),
which does the actual task of gathering. The implementation of `GatherFunctor`
is in fact massively complex for something that ultimately just calls `memcpy`
in a tight loop. There are a few reasons for that:

- In `GatherFunctorCPU`, they're specialising on the width of integers needed for
  indices (`int32` vs `int64`), and also compiling two special fast paths for
  slice elements of size 10 & 20 (why those sizes? IDK). That explains basically
  all the complexity in `GatherFunctor` (which should really just be a call to
  `HandleCopies`), and also explains some of the extra template params passed to
  `HandleCopies`.
- All the code needs to use abstract ops to deal with tensors because it can be
  template-specialised into either GPU or CPU code (!!). The GPU implementation
  bypasses the complications of `GatherFunctorCPU`, so the comments above don't
  apply to it.
- `HandleCopies` needs to have a C++ lambda do the actual work of copying
  because it supports sharding of computation across CPUs/GPUs with `Shard`.
  Incidentally, that class is deprecated in favour of the
  [`threadpool.h`](https://github.com/tensorflow/tensorflow/blob/1d8d2a0501e619a7a5f8ee94b18c4d93e243fd1c/tensorflow/core/lib/core/threadpool.h),
  according to a recent comment on the impl of `Shard`.
- `HandleCopies` is also complicated by the use of [software-based cache
  prefetching](https://en.wikipedia.org/wiki/Cache_prefetching), which I didn't
  even realise was a thing you could do on x86 CPUs (can you also give branch
  predictor hints, I wonder?).
- `HandleCopies` is also complicated by the fact that it handles tensors of
  strings, which require Eigen calls instead of just `memcpy`s (ugh).
- There's some bizarre shit I don't understand with applying `SubtleMustCopy` to
  input indices (which adds a volatile qualifier to the pointer, or something
  like that). Why would that ever be necessary? Surely the values of `indices`
  cannot ever change after the start of the `GatherFunctor`?
- Concurrency also introduces a small amount of code for locking, and so on. All
  in all, the thing is just very complicated.
  
With the possible exception of concurrency, and _maybe_ vectorisation (if not
supplied by `memcpy`), I think I can ignore basically all of those issues.
Really, just reshaping the input tensor to something reasonable & getting direct
access to the data will probably suffice for my purposes (but preclude running
on a GPU).

# Will this be extensible to multi-problem case?

Short answer: no, I'm probably not going to bother. Given that vectorisation
doesn't help very much on CPU, I think I'm better off implementing just the
single-problem case, and _maybe_ adding some concurrency (with TF thread pools)
to make it run a bit faster on machines with many CPUs available. Also, it may
be worth "lifting" my ASNet implementation so that it takes a connectivity
pattern as an input, then using
[`map_fn`](https://www.tensorflow.org/api_docs/python/tf/map_fn) and [`ragged
placeholders`](https://www.tensorflow.org/api_docs/python/tf/ragged/placeholder)
to make things work.
