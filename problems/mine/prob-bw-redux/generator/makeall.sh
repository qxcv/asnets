#!/bin/bash

set -e

make

mkdir -p ../problems

for n in 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50; do
    for i in {1..10}; do
        # seed is combination of n and i
        seed=$(($n*100+$i))
        # "es" = "extended seed", to disambiguate from original seeds used for
        # AAAI
        ./blocksworld "n${n}_es${i}_r${seed}" `./bwstates -s 2 -n $n -r $seed` | tail -n +49 > ../problems/prob_bw_n${n}_es${i}.pddl || break
    done
done
