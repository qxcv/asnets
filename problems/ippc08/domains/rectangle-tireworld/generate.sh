#! /bin/bash

export PATH=../../generators/rectangle-tireworld/:$PATH

cp ../../generators/rectangle-tireworld/domain.pddl ./

#usage: generate-RTW [options]
#options:
#  -x x, --xsize=x       width of the rectangle world
#  -y y, --ysize=y       height of the rectangle world
#  -h h, --horizontal=h  number of horizontal lines in the rectangle world
#  -v v, --vertical=h    number of vertical lines in the rectangle world
#  -u u, --unsafe=u      number of unsafe points in the rectangle world
#  -s s, --seed=s        random number generator seed (if not using time)
#  -?     --help         display this help and exit

function make-rtw {
   generate-RTW -x $2 -y $3 -h $4 -v $5 -u $6 -s $7 \
   > $1-x$2-y$3-h$4-v$5-u$6-s$7.pddl
}

#cmd     pXX  x  y  h  v   u  s
make-rtw p01  5  5  2  2   0  1
make-rtw p02  5  5  2  3  15  2
make-rtw p03  7  7  4  3   0  3
make-rtw p04  7  7  3  4  20  4
make-rtw p05  9  9  3  5  30  5
make-rtw p06 11 11  4  3  40  6
make-rtw p07 11 11  3  5  80  7
make-rtw p08 11 11  6  7  80  8
make-rtw p09 15 15  4  4  60  9
make-rtw p10 15 15 10 10 150 10
make-rtw p11 20 20  5  5  80 11
make-rtw p12 20 20 15 15 300 12
make-rtw p13 30 30  8  8 100 13
make-rtw p14 30 30 25 20 700 14
make-rtw p15 60 60 15 25 1500 15

rm -f generate.sh~