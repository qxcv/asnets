This is a new set of problems for the probabilistic blocksworld domain from
IPPC'08. The original set of problems was a bit too small for evaluation, and
also used a "(:goal-reward)" directive which MDPSim doesn't support properly (I
suspect it has a per-domain goal reward or something, and tries to set the same
goal reward for EVERY problem from a given domain every time it encounters that
directive!).

problems/ contains some problems with random seeds. aaai-orig-probs/ contains
problems for results reported in original AAAI submission. I was worried there
wouldn't be enough in the original set (one of the reviewer's concerns) and that
seed generation could have been flawed, so I regenerated the instances in
problems/ for the final.
