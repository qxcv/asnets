#!/bin/bash
rm -f jobs.txt

add_run() {
    echo ray submit --tmux ./ray-cluster/ray-cluster-cfg.yaml \
         ./ray-cluster/_cluster_run_det_baseline.py \
        --args="'--ray-connect localhost:6379 $@'" >> jobs.txt
}

for domain in det_gold_miner det_matching_bw_custom det_blocksworld; do
     add_run "experiments.${domain}"
done
shuf < jobs.txt > jobs-2.txt
mv jobs-2.txt jobs.txt
echo "Got $(wc -l jobs.txt) jobs; submitting to cluster"
cat jobs.txt | while read job; do
   echo -e "\n\n\n\n\n\n"
   echo "Submitting $job"
   sh -c "$job" < /dev/null \
       || (echo "Got retcode $?, stopping loop" && break)
done
