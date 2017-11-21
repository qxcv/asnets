#!/usr/bin/env python3
"""Code for loading networks in a shell. It turns out that loading networks is
harder than it should be because I don't have a very good configuration/weight
interface :("""

import joblib
import rpyc
from sandbox.rocky.tf.policies.categorical_mlp_policy import \
    CategoricalMLPPolicy

from fpg import get_problem_names, SingleProblem
from models import PropNetwork
from multiprob import ProblemServer
from supervised import ProblemServiceConfig, PlannerExtensions
from rllab_interface import create_environment

# we always want this to be true if we're going to do interactive stuff!
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

# TODO: aside from testing, I still need to do the following bits to get the
# visualisation I want:
#
# - Write some code to extract a trajectory of visited states.
# - Figure out how to pull intermediate activations out of the network. Right
#   now it can't really do it. Need to explicitly add some ops to network to
#   pull them out, perhaps.
# - Actually write vis code.


class InstantiatedNetwork:
    def __init__(self, instantiator, policy, flat_env, single_problem):
        self.instantiator = instantiator
        self.flat_env = flat_env
        self.single_problem = single_problem
        self.policy = policy

    def extract_goal_trajectory(self):
        """Pull out a trajectory which heads straight to goal, with no repeated
        states (and ideally decreasing V(s), which I'll hopefully figure out
        how to do without diving into extension code)."""
        pass


class NetworkInstantiator:
    """Loads a previously-trained WeightManager for a domain, then lets you
    instantiate a bunch of PropNetworks and Environments for specific
    problems."""

    def __init__(self,
                 snapshot_path,
                 use_lm_cuts=True,
                 extra_ppddl=(),
                 heuristic_data_gen_name='h-add',
                 norm_response=False):
        self.snapshot_path = snapshot_path
        self.weight_manager = joblib.load(snapshot_path)
        # we always use these files when loading data
        self.extra_ppddl = list(extra_ppddl)
        self.heuristic_data_gen_name = heuristic_data_gen_name
        self.norm_response = norm_response
        self.use_lm_cuts = use_lm_cuts

    def net_for_problem(self,
                        problem_ppddl_paths,
                        prob_name=None,
                        dropout=0.0):
        """Instantiate a network and environment for a specific problem. Also
        creates a handler for the problem (maybe this is a bad idea? IDK)."""
        all_ppddl_paths = self.extra_ppddl + list(problem_ppddl_paths)
        if prob_name is None:
            # get it automatically (and slowly…)
            prob_name = get_problem_names(all_ppddl_paths)[0]
        service_config = ProblemServiceConfig(
            all_ppddl_paths,
            prob_name,
            heuristic_name=self.heuristic_data_gen_name,
            # doesn't matter, use a cheap one
            teacher_heur='h-add',
            use_lm_cuts=self.use_lm_cuts)
        problem_server = ProblemServer(service_config)
        single_problem = SingleProblem(prob_name, problem_server)
        planner_exts = PlannerExtensions(service_config.pddl_files,
                                         service_config.init_problem_name)
        raw_env, flat_env = create_environment(
            single_problem.prob_meta,
            planner_exts,
            heuristic_name=self.heuristic_data_gen_name)
        problem_network = PropNetwork(
            self.weight_manager,
            single_problem.prob_meta,
            dropout=dropout,
            norm_response=self.norm_response)
        # env_spec is SingleProblem.env_spec
        policy = CategoricalMLPPolicy(
            env_spec=single_problem.env_spec,
            prob_network=problem_network,
            name='policy')
        # returns policy, RLLab environment, problem server handles
        return InstantiatedNetwork(self, policy, flat_env, single_problem)
