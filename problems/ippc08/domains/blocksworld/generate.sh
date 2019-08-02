#! /bin/bash

export PATH=../../generators/blocksworld/:$PATH

# DON'T DO THAT ! -> cp ../../generators/blocksworld/domain.pddl ./

function make-bw {
   blocksworld -c $2 -C $3 -g $4 $1 `bwstates -s 2 -n $5` \
   > $1-c$2-C$3-g$4-n$5.pddl
}

#cmd    pXX  c  C  g  n
make-bw p01  0  0  1  5
make-bw p02  1  1 20  5
make-bw p03  1  2 40  5
make-bw p04  2  1  0  5
make-bw p05  0  0  1 10
make-bw p06  1  1 20 10
make-bw p07  1  2  0 10
make-bw p08  3  2  0 10
make-bw p09  0  0  1 14
make-bw p10  1  1 20 14
make-bw p11  1  2  0 14
make-bw p12  3  2  0 14
make-bw p13  0  0  1 18
make-bw p14  1  1 20 18
make-bw p15  3  2  0 18

rm -f generate.sh~
