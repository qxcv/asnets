#! /bin/bash

export PATH=../../generators/zenotravel/:$PATH

cp ../../generators/blocksworld/domain.pddl ./

#usage: zeno [--cities <n>] [--persons <n>] [--aircrafts <n>] [--seed <seed>]
function make-zt {
   zeno --cities $2 --persons $3 --aircrafts $4 --seed $5 \
   > $1-c$2-p$3-a$4-s$5.pddl
}

#cmd    pXX  c  p  a  s
make-zt p01  4  2  2  3846
make-zt p02  5  2  2  17462
make-zt p03  5  5  3  3674
make-zt p04  6  2  2  12861
make-zt p05  6  5  3  24056
make-zt p06  7  5  3  6554
make-zt p07  7 10  6  24564
make-zt p08  8  5  3  27436
make-zt p09  9 10  6  29223
make-zt p10 10  5  3  15832
make-zt p11 11 10  6  21350
make-zt p12 13  5  3  14893
make-zt p13 14 10  6  12510
make-zt p14 15 10  6  11709
make-zt p15 20 10  6  24164

#make-zt p01  6  2  2  3846
#make-zt p02  7  2  2  17462
#make-zt p03  8  2  2  3674
#make-zt p04  9  2  2  12861
#make-zt p05 10  2  2  24056
#make-zt p06 11  5  3  6554
#make-zt p07 12  5  3  24564
#make-zt p08 13  5  3  27436
#make-zt p09 14  5  3  29223
#make-zt p10 15  5  3  15832
#make-zt p11 16 10  6  21350
#make-zt p12 17 10  6  14893
#make-zt p13 18 10  6  12510
#make-zt p14 19 10  6  11709
#make-zt p15 20 10  6  24164


rm -f generate.sh~
