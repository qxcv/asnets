#! /bin/bash

export PATH=../../generators/search-and-rescue/:$PATH

cp ../../generators/search-and-rescue/search-and-rescue-domain.pddl ./domain.pddl

#usage: generate-SnR [options]
#options:
#  -z z, --zones=z       number of zones
#  -?     --help         display this help and exit


function make-SnR {
   generate-SnR -z $2 \
   > $1-z$2.pddl
}

rm -f p*.pddl

#cmd     pXX  z
make-SnR p01  4
make-SnR p02  5
make-SnR p03  6
make-SnR p04  8
make-SnR p05 10
make-SnR p06 12
make-SnR p07 14
make-SnR p08 16
make-SnR p09 20
make-SnR p10 25
make-SnR p11 30
make-SnR p12 35
make-SnR p13 40
make-SnR p14 45
make-SnR p15 50

rm -f generate.sh~
