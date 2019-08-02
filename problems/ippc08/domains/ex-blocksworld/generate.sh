#! /bin/bash

export PATH=../../generators/ex-blocksworld/:$PATH

cp ../../generators/ex-blocksworld/domain.pddl ./

function make-bw {
#./ex-blocksworld <name-suffix> <blocks-in-goal> `./bwstates -s 2 -n <#-blocks> -r <seed>`
    ex-blocksworld $1 $2 `bwstates -s 2 -n $3 -r $4` \
	> $1-n$2-N$3-s$4.pddl
}

#cmd    pXX  n  N  s
make-bw p01  2  5  1
make-bw p02  3  5  2
make-bw p03  3  6  3
make-bw p04  4  6  4
make-bw p05  5  7  5
make-bw p06  6  8  6
make-bw p07  7  9  7
make-bw p08  8 10  8
make-bw p09  9 11  9
make-bw p10 10 12 10
make-bw p11 11 13 11
make-bw p12 12 14 12
make-bw p13 13 15 13
make-bw p14 14 16 14
make-bw p15 15 17 15

rm -f generate.sh~