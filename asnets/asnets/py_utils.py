"""Generic utilities that don't depend on anything outside of the Python stdlib
& standard numeric libs, and are all pure Python."""

from contextlib import contextmanager
from json import dumps
from time import time
from weakref import proxy, ProxyTypes

import ctypes
import random
import struct
import sys

import numpy as np


def weak_ref_to(obj):
    """Create a weak reference to object if object is not a weak reference. If
    object is a weak reference, then return that reference unchanged."""
    if obj is None or isinstance(obj, ProxyTypes):
        return obj
    return proxy(obj)


def strip_parens(thing):
    """Convert string of form `(foo bar baz)` to `foo bar baz` (i.e. strip
    leading & trailing parens). More complicated than it should be b/c it does
    safety checks to catch my bugs :)"""
    assert len(thing) > 2 and thing[0] == "(" and thing[-1] == ")", \
        "'%s' does not look like it's surrounded by parens" % (thing,)
    stripped = thing[1:-1]
    assert "(" not in stripped and ")" not in stripped, \
        "parens in '%s' aren't limited to start and end" % (thing,)
    return stripped


def weighted_batch_iter(arrays, weights, batch_size, n_batches):
    assert len(arrays) > 0, "no arrays given (?)"
    assert len(weights) > 0, "no weights given (?)"
    weight_sum = np.sum(weights)
    assert weight_sum > 1e-7, "weights tiny (?)"
    probs = weights / np.sum(weights)
    for _ in range(n_batches):
        chosen_inds = np.random.choice(
            len(probs), size=(batch_size, ), replace=True, p=probs)
        yield tuple(a[chosen_inds] for a in arrays)


class TimerContext:
    """Keeps track of the average time taken to perform each instance of a
    repeated operation (e.g forward/back propagation, planning on a problem
    state, etc.). Recorded mean times can easily be printed as JSON."""
    def __init__(self):
        # keep running mean & count instead of running sum & count (probably
        # works better when you have MANY events & don't want to do big-ish
        # multiplications)
        self._counts = dict()
        self._means = dict()

    @contextmanager
    def time_block(self, timer_name):
        """Context manager that measures elapsed time upon exit & adds it to
        the averages for timer_name."""
        start_time = time()
        try:
            yield
        finally:
            elapsed = time() - start_time
            old_count = self._counts.setdefault(timer_name, 0)
            old_mean = self._means.setdefault(timer_name, 0)
            new_count_f = old_count + 1.0
            new_mean = old_mean * (old_count / new_count_f) \
                + elapsed / new_count_f
            self._counts[timer_name] += 1
            self._means[timer_name] = new_mean

    def to_dict(self):
        return dict(self._means)

    def to_json(self):
        return dumps(self._means, sort_keys=True, indent=2)


def set_c_seeds(seed):
    """Set C stdlib seeds (with srand, srand48, etc.). Those generators are
    used by MDPSim, SSiPP, et al., so setting them is important."""
    assert sys.platform == 'linux', \
        "this fn only works on Linux right now"
    libc = ctypes.CDLL("libc.so.6")
    # srand() takes a normal int
    srand_seed = ctypes.c_int(seed)
    # srand48() takes long int
    srand48_seed = ctypes.c_long(seed)
    # the *rand48*() functions need a three-short array (as in, sequences of
    # three int16s)
    ushorts = struct.unpack('<HHH', struct.pack('<q', seed)[:6])
    arr_type = ctypes.c_ushort * 3
    seed48_seed = arr_type(*ushorts)

    # set types (this doesn't seem to be necessary, but I guess offers some
    # type checking)
    libc.srand.argtypes = (ctypes.c_int,)
    libc.srand48.argtypes = (ctypes.c_long,)
    libc.seed48.argtypes = (arr_type,)

    # now set seeds!
    libc.srand(srand_seed)
    libc.srand48(srand48_seed)
    libc.seed48(seed48_seed)


def set_random_seeds(seed):
    """Set random seeds that are relevant for main process."""
    print(f"Setting C/Python/Numpy seeds to {seed}")
    set_c_seeds(seed)
    random.seed(seed)
    np.random.seed(seed)
    if 'tf' in globals() or 'tensorflow' in sys.modules:
        print(f"Setting TF seed to {seed}")
        tf = sys.modules['tensorflow']
        tf.random.set_seed(seed)
    else:
        print(f"Skipping TF RNG seeding")


def find_tail_cycle(item_list, max_cycle_len=3):
    """Dumb brute force thing for finding cycle with maximum number of repeats
    at tail end of a sequence of things. Useful for trimming long lists of
    actions that have cycles at the end."""
    n = len(item_list)
    max_chunk_size = 0
    max_repeats = 0
    max_cycle_len = min(max_cycle_len, min(n, np.ceil(n / 2.0)))
    for chunk_size in range(1, max_cycle_len):
        chunk = item_list[-chunk_size:]
        repeats = 1
        for backoff in range(1, n // len(chunk)):
            start_idx = n - len(chunk) * backoff
            if item_list[start_idx:start_idx + len(chunk)] != chunk:
                break
            repeats = backoff
        if repeats > 1:
            max_chunk_size = chunk_size
            max_repeats = repeats
            break
    if max_chunk_size > 0:
        repeat_tail = chunk * max_repeats
        assert item_list[-len(repeat_tail):] == repeat_tail, \
            (repeat_tail, item_list[-len(repeat_tail):])
        return chunk, repeats
    return [], 0


def remove_cycles(item_list, max_cycle_len=3, max_cycle_repeats=3):
    """Removes long cycles of repeated elements from the end of a list. Looks
    for subsequence of elements at tail of list consisting of up to
    `max_cycle_len` elements that are repeated at least `max_cycle_repeats`
    times` (e.g 2 elements repeated 10 times). Will remove all but
    `max_cycle_repeats` instances of the cycle. Does nothing if no sufficiently
    large cycles are found."""
    tail_cycle_vals, tail_cycle_repeats = find_tail_cycle(item_list)
    repeat_delta = tail_cycle_repeats - max_cycle_repeats
    num_removed = 0
    if repeat_delta > 0:
        # if the cycle at the end is repeated too many times, remove it
        num_removed = len(tail_cycle_vals) * repeat_delta
        item_list = item_list[:-num_removed]
    return item_list, num_removed
