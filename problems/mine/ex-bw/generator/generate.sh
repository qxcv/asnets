#!/bin/bash

# ensure we have the thingos
make

# copied from William's generator script (usage: makeprob $family $blocks $seq)
function makeprob {
    family=$1
    shift
    blocks=$1
    shift
    seq=$1
    shift
    name="$(printf "ex-bw-%s-n%02d-s%02d" "${family}" "${blocks}" "${seq}")"
    seed_hex="$(echo "$name" | md5sum | cut -d " " -f 1 | head -c 10)"
    seed="$(echo $((16#$seed_hex)) | head -c 6)"
    echo "$seed_hex" "$seed"
    name="$(printf "%s-r%06d" "${name}" "${seed}")"
    # Usage:
    #     ./ex-blocksworld <name-suffix> <blocks-in-goal> \
    #         `./bwstates -s 2 -n <blocks-in-state> -r <seed>`
    # (blocks-in-goal can be < blocks-in-state if you want)
    dest="../${family}/${name}.pddl"
    echo "Writing to $dest"
    mkdir -pv "$(dirname "$dest")"
    ./ex-blocksworld "${name}" "${blocks}" \
        $(./bwstates -s 2 -n "${blocks}" -r "${seed}") > "$dest"
}

# training problems (60 of them; I'll have to select tractable subset)
for blks in 4 5 6 7 8 9; do
    for seq in {0..9}; do
        makeprob train $blks $seq
    done
done

# testing problems (no idea how to trim these down to only solvable ones)
for blks in {11..20}; do
    for seq in {0..2}; do
        makeprob test $blks $seq
    done
done
