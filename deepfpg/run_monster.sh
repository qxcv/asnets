#!/bin/bash

# re-create monster experiment from thesis
# (this depends on my patch that disables L2 regularisation, etc.)

for i in {1..5}; do
    out_name=monster-out-${i}.txt
    ./run_experiment.py experiments.actprop_${i}l_h_add_no_lmcut experiments.monster > $out_name 2> $out_name &
    sleep 2
done
echo "Waiting for jobs to finish (this probably takes ~15 minutes). Check monster-out-*.txt for progress."
wait
