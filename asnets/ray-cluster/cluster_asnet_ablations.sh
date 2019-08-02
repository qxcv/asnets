#!/bin/bash

# Run ablated experiments on cluster, minus experiments that were already run
# before as aprt of main set

rm -f jobs.txt

add_run() {
    echo ray submit --tmux ./ray-cluster/ray-cluster-cfg.yaml \
         ./ray-cluster/_cluster_run_experiment.py \
        --args="'--job-ncpus 1 --enforce-job-ncpus --ray-connect localhost:6379 $@'" >> jobs.txt
}

for domain in triangle_tireworld cosanostra ex_blocksworld prob_blocksworld; do
    for model in actprop_2l_h_add{_no_skip,_no_history,_no_lmcut} actprop_{1,3}l_h_add; do
         add_run "experiments.${model}" "experiments.${domain}"
    done
done
for domain in det_gold_miner det_matching_bw_custom det_blocksworld; do
    for model in actprop_2l_fd{_no_skip,_no_history,_no_lmcut} actprop_{1,3}l_fd; do
         add_run "experiments.${model}" "experiments.${domain}"
    done
done
shuf < jobs.txt > jobs-2.txt
mv jobs-2.txt jobs.txt
echo "Got $(wc -l jobs.txt) jobs; submitting to cluster"
cat jobs.txt | while read job; do
   echo -e "\n\n\n\n\n\n"
   echo "Submitting $job"
   # the </dev/null is necessary to stop this from breaking our "while read"
   # loop, per https://stackoverflow.com/a/13800476
   sh -c "$job" < /dev/null \
       || (echo "Got retcode $?, stopping loop" && break)
done
