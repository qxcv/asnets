#!/bin/bash

# usage: matching-bw-generator.sh <base name> <n>

./bwstates -n $2 > temp.blocks
./bwstates -n $2 >> temp.blocks

./2pddl-typed -d temp.blocks -n $2 > $1-typed.pddl
./2pddl-untyped -d temp.blocks -n $2 > $1-untyped.pddl

rm -f temp.blocks
