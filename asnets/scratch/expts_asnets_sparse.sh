#!/bin/bash

run_expt() {
    mod="$1"
    dom="$2"
    ./run_experiment experiments.${mod} experiments.${dom} \
                     2>&1 | tee out_sparse_${dom}.txt  &
}

run_expt actprop_2l_h_add_sparse cosanostra_345
# run_expt actprop_2l_h_add_sparse prob_blocksworld
run_expt actprop_2l_h_add_sparse_no_lmcut_or_history triangle_tireworld_23
echo "Started, waiting for jobs"
wait

# Commands for plotting CosaNostra frames:
# ./sparse_activation_vis ./sparse/cn-weights.pkl ./sparse/cn-domain.pddl \
#   ./sparse/cn-problem.pddl --use-lm-cut --save-dir frames/cn \
#   --no-draw-action --draw-act-seq --ext .pdf
#
# PBW frames, also uses LM-cut:
# ./sparse_activation_vis ./sparse/pbw-weights.pkl ./sparse/pbw-domain.pddl \
#   ./sparse/pbw-problem.pddl --use-lm-cut --save-dir frames/pbw \
#   --no-draw-action --draw-act-seq --ext .pdf
#
# TTW frames, generally trained without LM-cut
# ./sparse_activation_vis ./sparse/ttw-weights.pkl ./sparse/ttw-domain.pddl \
#   ./sparse/ttw-problem.pddl --save-dir frames/ttw \
#   --no-draw-action --draw-act-seq --ext .pdf
