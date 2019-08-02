#!/bin/bash

# Generate custom training problems for gold-miner. These ones are a little bit
# larger.

set -e

DEST_DIR="../mine/"
DEST_DIR_TRAIN="${DEST_DIR}train/"
DEST_DIR_TEST="${DEST_DIR}test/"

make
mkdir -p "${DEST_DIR}" "${DEST_DIR_TRAIN}"
mkdir -p "${DEST_DIR}" "${DEST_DIR_TEST}"

cp -v ../learning/gold-miner-typed.pddl "${DEST_DIR}/domain.pddl"

# gen <size> <seed>
# (for generating training problems)
gen() {
    size="$1"
    seed="$2"
    dest_name="gm-${size}x${size}-s${seed}.pddl"
    echo "Generating training problem $dest_name"
    # prepend/append some 9s to the seed so we don't collide with test problems
    seed="99${seed}99"
    ./gold-miner-generator -c "$size" -r "$size" -s "${seed}" > "${DEST_DIR_TRAIN}/${dest_name}"
}

# gen_test <size> <seed>
# (for generating test problems; different naming)
gen_test() {
    size="$1"
    seed="$2"
    size_label=$size
    if [ $size -lt 10 ]; then
        size_label=0$size
    fi
    dest_name="gm-${size_label}x${size_label}-s${seed}-big.pddl"
    echo "Generating test problem $dest_name"
    ./gold-miner-generator -c "$size" -r "$size" -s "$seed" > "${DEST_DIR_TEST}/${dest_name}"
}

# Figures below indicate how long each problem took to solve with a single round
# of LRTDP (with h-add heuristic, in SSiPP implementation). Costs are
# approximate---varies a bit from run to run.

# 4x4 problems
gen 4 40  # solvable in ~4s, cost ~17
gen 4 41  # solvable in ~6s, cost ~16
gen 4 42  # solvable in ~1s, cost ~14
gen 4 43  # solvable in ~15s, cost ~21
gen 4 44  # solvable in ~1s, cost ~13

# 5x5 problems
gen 5 50  # unk
gen 5 51  # unk
gen 5 52  # unk
gen 5 53  # unk
gen 5 54  # unk
gen 5 55  # unk
gen 5 56  # unk
gen 5 57  # unk
gen 5 58  # unk
gen 5 59  # unk

# 6x6 problems
gen 6 60  # unk
gen 6 61  # unk
gen 6 62  # unk
gen 6 63  # unk
gen 6 64  # unk
gen 6 66  # unk
gen 6 67  # unk
gen 6 68  # unk
gen 6 69  # unk

# 7x7 problems (full size!)
gen 7 70  # unk
gen 7 71  # unk
gen 7 72  # unk
gen 7 73  # unk
gen 7 74  # unk
gen 7 75  # unk
gen 7 76  # unk
gen 7 77  # unk
gen 7 78  # unk
gen 7 79  # unk

# 8x8, 9x9, 10x10, and 11x11 problems (HUUUUUUGE)
gen 8 80  # unk
gen 8 81  # unk
gen 8 82  # unk
gen 8 83  # unk
gen 8 84  # unk
gen 9 90  # unk
gen 9 91  # unk
gen 9 92  # unk
gen 9 93  # unk
gen 9 94  # unk
gen 10 100  # unk
gen 10 101  # unk
gen 10 102  # unk
gen 10 103  # unk
gen 10 104  # unk
gen 11 110  # unk
gen 11 111  # unk
gen 11 112  # unk
gen 11 113  # unk
gen 11 114  # unk


###########################
# HARD TEST PROBLEMS
###########################

# ~15 bigger eval tasks that LAMA will hopefully choke on (& ASNets will
# hopefully NOT choke on).
for size in 7 8 9 10 13 16 19; do
    for last_dig in {0..2}; do
        gen_test ${size} ${size}${last_dig}
    done
done
