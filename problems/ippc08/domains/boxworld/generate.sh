#! /bin/bash

export PATH=../../generators/boxworld/:$PATH

#usage: ./boxworld [-h]
#                  [-b box-count]
#                  [-c city-count]
#                  [-dc drive-cost]
#                  [-fc fly-cost]
#                  [-dr delivery-reward]
#                  [-tlc truck-load-cost]
#                  [-tuc truck-unload-cost]
#                  [-plc plane-load-cost]
#                  [-puc plane-unload-cost]
#                  [-gr goal-reward]
#                  [-dn domain name]
#                  [-pn problem name]

function make-bw {
   boxworld -pn box-$1 -b $2 -c $3 -dc $4 -fc $5 -dr $6 -gr $7 \
   > $1-b$2-c$3-dc$4-fc$5-dr$6-gr$7.pddl
}

#cmd    pXX  b  c dc fc  dr  gr
make-bw p01 10  5  0  0   0   1
make-bw p02 10  5  0  0   1  10
make-bw p03 10  5  5 25  50 500
make-bw p04 10 10  0  0   0   1
make-bw p05 10 10  8 25  50 500
make-bw p06 10 15  0  0   0   1
make-bw p07 10 15  5 30  50 500
make-bw p08 15 10  0  0   0   1
make-bw p09 15 10  5 25 100 500
make-bw p10 15 15  0  0   0   1
make-bw p11 15 15  0  0   1   0
make-bw p12 15 15  5 25 100 500
make-bw p13 20 20  0  0   0   1
make-bw p14 20 20  0  0   1  20
make-bw p15 20 20  5 25 100 500

rm -f generate.sh~
