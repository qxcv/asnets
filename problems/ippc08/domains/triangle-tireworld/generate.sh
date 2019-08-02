#! /bin/bash

export PATH=../../generators/triangle-tireworld/:$PATH

cp ../../generators/triangle-tireworld/domain.pddl ./

function make-tt {
   extract-problem.sh $2 $3 ../../generators/triangle-tireworld/gen-triangle.pddl \
   > $1.pddl
}

#cmd    pXX from to
make-tt p01 18 23
make-tt p02 24 29
make-tt p03 30 35
make-tt p04 36 41
make-tt p05 42 47
make-tt p06 48 53
make-tt p07 54 59
make-tt p08 60 65
make-tt p09 66 71
make-tt p10 72 76

rm -f generate.sh~
