#!/bin/bash

# Runs some forward- and back-prop benchmark on ASNets. I've used this to test
# the performance impact of my custom, ASNet-specific TensorFlow ops.

set -e

BENCH_DIR="./mbw-bench/"
SAVED_DATA_PATH="./mbw-bench/saved_data.pkl"
USAGE="USAGE: $0 <core-id>"
mkdir -p "$BENCH_DIR"

do_train() {
    # Train script. Don't think I'll use this directly, but do want to have it
    # around in case I forget what it's meant to do :P
    DOMAIN_PATH="../problems/ipc08-learn/matching-bw/learning/matching-bw-typed.pddl"
    PROB_DIR="../problems/ipc08-learn/matching-bw/mine/train/"
    ./run_asnets -e "$BENCH_DIR/train-run/" -O "num_layers=2,hidden_size=16" \
        --det-eval --l2-reg 0.001 --l1-reg 0.0 -R 1 -m actprop -L 300 -t 7200 \
        --supervised-lr 0.01 --supervised-bs 128 --supervised-early-stop 0 \
        --save-every 1 --ssipp-teacher-heur lm-cut --opt-batch-per-epoch 1000 \
        --teacher-planner fd --sup-objective ANY_GOOD_ACTION --lr-step 2 0.001 \
        --lr-step 10 0.0001 --save-training-set "$SAVED_DATA_PATH" \
        "$DOMAIN_PATH" "${PROB_DIR}/mbw-b5-t1-s20.pddl" \
        "${PROB_DIR}/mbw-b5-t2-s21.pddl" "${PROB_DIR}/mbw-b5-t3-s22.pddl" \
        "${PROB_DIR}/mbw-b8-t1-s0.pddl" "${PROB_DIR}/mbw-b8-t1-s1.pddl" \
        "${PROB_DIR}/mbw-b8-t1-s2.pddl" "${PROB_DIR}/mbw-b8-t1-s3.pddl" \
        "${PROB_DIR}/mbw-b8-t2-s10.pddl" "${PROB_DIR}/mbw-b8-t3-s11.pddl" \
        "${PROB_DIR}/mbw-b8-t4-s12.pddl" "${PROB_DIR}/mbw-b8-t5-s13.pddl" \
        "${PROB_DIR}/mbw-b9-t1-s31.pddl" "${PROB_DIR}/mbw-b9-t2-s32.pddl" \
        "${PROB_DIR}/mbw-b9-t3-s33.pddl" "${PROB_DIR}/mbw-b9-t4-s34.pddl"
}


do_test() {
    # Test script. Can be timed externally. Main difference from above is the
    # use of a --use-saved-training-set flag instead of a --save-training-set
    # flag.
    DOMAIN_PATH="../problems/ipc08-learn/matching-bw/learning/matching-bw-typed.pddl"
    PROB_DIR="../problems/ipc08-learn/matching-bw/mine/train/"
    out_dir="${BENCH_DIR}/test-run-$(date -Iseconds)"
    echo "Will store things in ${out_dir}"
    mkdir -pv "$out_dir"
    /usr/bin/time -v -o "${out_dir}/time.txt" taskset -c "$core" \
        ./run_asnets -e "${out_dir}/test-run/" \
            -O "num_layers=2,hidden_size=16" --det-eval --l2-reg 0.001 \
            --l1-reg 0.0 -R 1 -m actprop -L 300 -t 3600 --supervised-lr 0.01 \
            --supervised-bs 128 --supervised-early-stop 0 --save-every 1 \
            --ssipp-teacher-heur lm-cut --opt-batch-per-epoch 1000 \
            --teacher-planner fd --sup-objective ANY_GOOD_ACTION \
            --lr-step 2 0.001 --lr-step 10 0.0001 \
            --use-saved-training-set "$SAVED_DATA_PATH" \
            "$DOMAIN_PATH" "${PROB_DIR}/mbw-b5-t1-s20.pddl" \
            "${PROB_DIR}/mbw-b5-t2-s21.pddl" "${PROB_DIR}/mbw-b5-t3-s22.pddl" \
            "${PROB_DIR}/mbw-b8-t1-s0.pddl" "${PROB_DIR}/mbw-b8-t1-s1.pddl" \
            "${PROB_DIR}/mbw-b8-t1-s2.pddl" "${PROB_DIR}/mbw-b8-t1-s3.pddl" \
            "${PROB_DIR}/mbw-b8-t2-s10.pddl" "${PROB_DIR}/mbw-b8-t3-s11.pddl" \
            "${PROB_DIR}/mbw-b8-t4-s12.pddl" "${PROB_DIR}/mbw-b8-t5-s13.pddl" \
            "${PROB_DIR}/mbw-b9-t1-s31.pddl" "${PROB_DIR}/mbw-b9-t2-s32.pddl" \
            "${PROB_DIR}/mbw-b9-t3-s33.pddl" "${PROB_DIR}/mbw-b9-t4-s34.pddl"
    echo "Attempting to copy timing file"
    json_time="$(find "${out_dir}/test-run/" -type f -name 'timing.json' -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")"
    if [ -z "$json_time" ]; then
        echo "No timing.json in '$out_dir'?"
    else
        cp -v "$json_time" "$out_dir"
    fi
}


core="$1"
if [[ -z "$core" ]]; then
    echo "Error: invalid or missing core '$core'" >> /dev/stderr
    echo "$USAGE" >> /dev/stderr
    exit 1
fi
shift

if [[ ! ( -z "$@" ) ]]; then
    echo "Error: unprocessed arguments '$@'" >> /dev/stderr
    echo "$USAGE" >> /dev/stderr
    exit 1
fi

# To get timing.json from just train run (easy to adapt to test case):
# find mbw-bench/train-run/ -type f -name 'timing.json' -printf '%T@ %p\n' \
#     | sort -n | tail -1 | cut -f2- -d" "
echo "Running test for 1h"
do_test
echo "Test finished!"
