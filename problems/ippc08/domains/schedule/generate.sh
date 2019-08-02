#! /bin/bash

export PATH=../../generators/schedule/:$PATH

# example: schedule -c 2 -u 2 -l 1000

function make-s {
    schedule -c $2 -u $3 -l $4 \
   > $1-c$2-u$3-l$4.pddl
}

rm -f p*.pddl

#cmd   pXX  c  u    l
make-s p01  1  3   30
make-s p02  1  3   50
make-s p03  1  4  100
make-s p04  2  3   50
make-s p05  2  4  100
make-s p06  3  3  500
make-s p07  3  3  500
make-s p08  3  4  500
make-s p09  4  5 1000
make-s p10  4  5 1000
make-s p11  5  3 2000
make-s p12  5  3 2000
make-s p13  5  4 3000
make-s p14  7  5 3000
make-s p15 10  5 3000

rm -f generate.sh~