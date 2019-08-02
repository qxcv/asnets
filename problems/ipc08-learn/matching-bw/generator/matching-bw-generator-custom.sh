#!/bin/bash

# script usage: matching-bw-generator.sh (no args)

mkproblem() {
    dest_dir="$1"
    nblocks="$2"
    ntow="$3"
    # original seed passes straight through
    orig_seed="$4"
    pname="$5"
    # goal seed must be different
    goal_seed="$((orig_seed ^ 31282558))"
    pddl_dest="${dest_dir}/${pname}.pddl"
    echo -n "Making problem with pname=$pname, nblocks=$nblocks, ntow=$ntow,"
    echo -n " orig_seed=$orig_seed, goal_seed=$goal_seed"
    echo " --> $pddl_dest"
    ./bwkstates -s 1 -n "$nblocks" -s 1 -i "$ntow" -g "$ntow" -r "$orig_seed" > temp.blocks
    ./bwkstates -s 1 -n "$nblocks" -s 1 -i "$ntow" -g "$ntow" -r "$goal_seed" >> temp.blocks
    # don't use full states; we want to be compatible with IPC problems
    ./2pddl-typed -p "$pname" -f 0 -d temp.blocks -n "$nblocks" > "$pddl_dest"
    rm -f temp.blocks
}

mkdir -p ../mine/{train,test}

# args:    <dest-pddl>     <blocks>  <towers>  <seed>  <name>
# 5 blocks
mkproblem  ../mine/train/  5         1         20      mbw-b5-t1-s20
mkproblem  ../mine/train/  5         2         21      mbw-b5-t2-s21
mkproblem  ../mine/train/  5         3         22      mbw-b5-t3-s22
# 6 blocks
mkproblem  ../mine/train/  6         1         60      mbw-b6-t1-s60
mkproblem  ../mine/train/  6         2         61      mbw-b6-t2-s61
mkproblem  ../mine/train/  6         3         62      mbw-b6-t3-s62
mkproblem  ../mine/train/  6         2         63      mbw-b6-t3-s63
# 8 blocks
mkproblem  ../mine/train/  8         1         0       mbw-b8-t1-s0
mkproblem  ../mine/train/  8         1         1       mbw-b8-t1-s1
mkproblem  ../mine/train/  8         1         2       mbw-b8-t1-s2
mkproblem  ../mine/train/  8         1         3       mbw-b8-t1-s3
mkproblem  ../mine/train/  8         2         10      mbw-b8-t2-s10
mkproblem  ../mine/train/  8         3         11      mbw-b8-t3-s11
mkproblem  ../mine/train/  8         4         12      mbw-b8-t4-s12
mkproblem  ../mine/train/  8         5         13      mbw-b8-t5-s13
mkproblem  ../mine/train/  8         4         14      mbw-b8-t4-s14
mkproblem  ../mine/train/  8         3         15      mbw-b8-t3-s15
# 9 blocks. these will be REALLY expensive to solve!
mkproblem  ../mine/train/  9         1         31      mbw-b9-t1-s31
mkproblem  ../mine/train/  9         2         32      mbw-b9-t2-s32
mkproblem  ../mine/train/  9         3         33      mbw-b9-t3-s33
mkproblem  ../mine/train/  9         4         34      mbw-b9-t4-s34
mkproblem  ../mine/train/  9         3         35      mbw-b9-t3-s35
mkproblem  ../mine/train/  9         1         36      mbw-b9-t1-s36

# Test set contains three problems of each size in {8, 9, 10, 11, 12, 13, 14,
# 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}. First problem has a single tower,
# second problem has floor(sqrt(n)) towers, third problem has random number of
# towers chosen uniformly between 2 and floor(2*sqrt(n)).

for size in 8 9 10 11 12 13 14 15 20 25 30 35 40 45 50 55 60; do
    # seed RNG
    RANDOM="$((size ** 3))"
    echo "$RANDOM" > /dev/null
    max_size="$(echo "2 * sqrt(${size}.0) / 1 - 2" | bc)"
    for towers in 1 "$(echo "sqrt($size)" | bc)" "$((2 + RANDOM % max_size))"; do
        seed="$((RANDOM % 2048))"
        size_label=$size
        if [[ $size_label -lt 10 ]]; then
           # zero-pad
           size_label=0$size_label
        fi
        tower_label=$towers
        if [[ $tower_label -lt 10 ]]; then
            tower_label=0$tower_label
        fi
        pname="mbw-b$size_label-t$towers-s$seed"
        # args    <dest-pddl>     <blocks> <towers>  <seed>  <name>
        mkproblem "../mine/test/" "$size"  "$towers" "$seed" "$pname"
    done
done
