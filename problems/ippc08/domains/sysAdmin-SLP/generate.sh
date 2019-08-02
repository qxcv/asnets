#! /bin/bash

export PATH=../../generators/sysAdmin-SLP/:$PATH

cp ../../generators/sysAdmin-SLP/domain.pddl ./

#usage: generate-SA [options]
#options:
#  -n n, --nodes=n       number of nodes in the loop
#  -l l, --links=l       number of additionnal links
#  -s s, --seed=s        random number generator seed (if not using time)
#  -?     --help         display this help and exit

function make-SA {
    generate-SA -n $2 -l $3 -s $4 \
   > $1-n$2-l$3-s$4.pddl
}

rm -f p*.pddl

#cmd    pXX    n    l  s
make-SA p01    4    1  1
make-SA p02    5    2  2
make-SA p03    6    3  3
make-SA p04    8    4  4
make-SA p05   12    6  5
make-SA p06   16    8  6
make-SA p07   20   10  7
make-SA p08   25   12  8
make-SA p09   30   15  9
make-SA p10   60   30 10
make-SA p11  120   60 11
make-SA p12  240  120 12
make-SA p13  480  240 13
make-SA p14  960  480 14
make-SA p15 1920  960 15

rm -f generate.sh~
