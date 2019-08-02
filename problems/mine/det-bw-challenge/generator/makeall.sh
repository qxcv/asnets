#!/bin/bash

set -e

# build bwkstates
make
# common opts
DEST="../pddl/"
NPROBS_COMBINED=100
# 1501 because 6*1501=9006 (...because it's OVER 9000!)
NPROBS_STRAT18=1501
# 1000 because 10*10000=10000 & that seems large enough that Hector won't ask
# for "more experiments".
NPROBS_STRAT50=1000
COMMAND="python3 generator.py --write-domain "

# small test problems
# make 25-block problems
${COMMAND} --nprobs "$NPROBS_COMBINED" --seed 1428122178 25 "${DEST}"
# make 35-block problems
${COMMAND} --nprobs "$NPROBS_COMBINED" --seed 2107726020 35 "${DEST}"
# make 50-block problems
${COMMAND} --nprobs "$NPROBS_COMBINED" --seed 1184714140 50 "${DEST}"

# train problems
# make 5-block problems
${COMMAND} --nprobs 10 --seed 896255376 5  "${DEST}train"
# make 8-block problems
${COMMAND} --nprobs 10 --seed 236108287 8  "${DEST}train"
# make 8-block single-tower problems (to get long plans)
${COMMAND} --nprobs 10 --seed 270765476 --ntowers 1 8 "${DEST}train"
# make 9-block problems
${COMMAND} --nprobs 10 --seed 129483654 9  "${DEST}train"
# make 9-block single-tower problems
${COMMAND} --nprobs 10 --seed 129483654 --ntowers 1 9 "${DEST}train"
# make 10-block problems
${COMMAND} --nprobs 10 --seed 614849806 10 "${DEST}train"

# 7-block test problems w/o stratified sampling
${COMMAND} --nprobs 1000 --seed 627553762 7 "${DEST}test07"

# 18-block problems w/ stratified sampling over tower counts
# with 1 tower
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 281841352 --ntowers 1 18 "${DEST}strat18"
# with 3 towers
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 1831953420 --ntowers 3 18 "${DEST}strat18"
# with 5 towers
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 800935206 --ntowers 5 18 "${DEST}strat18"
# with 10 towers
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 1488094851 --ntowers 10 18 "${DEST}strat18"
# with 15 towers
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 1618215634 --ntowers 15 18 "${DEST}strat18"
# with 18 towers
${COMMAND} --nprobs "$NPROBS_STRAT18" --seed 585408346 --ntowers 18 18 "${DEST}strat18"

# 50-block problems w/ stratified sampling over tower counts
# with 1 tower
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 1612922803 --ntowers 1 50 "${DEST}strat50"
# with 4 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 378450903 --ntowers 4 50 "${DEST}strat50"
# with 7 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 795310950 --ntowers 7 50 "${DEST}strat50"
# with 10 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 45088672 --ntowers 10 50 "${DEST}strat50"
# with 15 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 353226086 --ntowers 15 50 "${DEST}strat50"
# with 20 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 1213313126 --ntowers 20 50 "${DEST}strat50"
# with 25 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 224502352 --ntowers 25 50 "${DEST}strat50"
# with 30 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 585020349 --ntowers 30 50 "${DEST}strat50"
# with 40 towers
${COMMAND} --nprobs "$NPROBS_STRAT50" --seed 1677989884 --ntowers 40 50 "${DEST}strat50"
