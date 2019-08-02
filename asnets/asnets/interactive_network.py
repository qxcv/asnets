#!/usr/bin/env python3
"""Code for loading networks in a shell. It turns out that loading networks is
harder than it should be because I don't have a very good configuration/weight
interface :("""

from collections import namedtuple

import joblib
import rpyc

from asnets.scripts.run_asnets import get_problem_names, SingleProblem
from asnets.models import PropNetwork
from asnets.multiprob import ProblemServer
from asnets.supervised import ProblemServiceConfig, PlannerExtensions

# we always want this to be true if we're going to do interactive stuff!
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

InstantiatedNetwork = namedtuple(
    'InstantiatedNetwork',
    ['instantiator', 'single_problem', 'policy', 'planner_exts'])


class NetworkInstantiator:
    """Loads a previously-trained WeightManager for a domain, then lets you
    instantiate a bunch of PropNetworks and Environments for specific
    problems."""

    def __init__(self,
                 snapshot_path,
                 use_lm_cuts=True,
                 use_history=True,
                 extra_ppddl=(),
                 heuristic_data_gen_name='h-add'):
        self.snapshot_path = snapshot_path
        self.weight_manager = joblib.load(snapshot_path)
        # we always use these files when loading data
        self.extra_ppddl = list(extra_ppddl)
        self.heuristic_data_gen_name = heuristic_data_gen_name
        self.use_lm_cuts = use_lm_cuts
        self.use_history = use_history

    def net_for_problem(self, problem_ppddl_paths, prob_name=None,
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
            use_lm_cuts=self.use_lm_cuts,
            use_act_history=self.use_history,
            teacher_planner='ssipp',
            random_seed=None)

        problem_server = ProblemServer(service_config)
        problem_server.service.initialise()

        single_problem = SingleProblem(prob_name, problem_server)

        policy = PropNetwork(
            self.weight_manager,
            single_problem.prob_meta,
            dropout=dropout,
            debug=False)

        # Normally we'd instantiate ProblemServer only in a master process,
        # and PlannerExtensions only in a subprocess that communicates with the
        # master. I'm instantiating both here so that it's possible to play
        # with & debug the PlannerExtensions class.
        #
        # WARNING: PlannerExtensions needs to instantiated last, otherwise
        # fully-initialised MDPSim & SSiPP will be visible in the child process
        # for ProblemServer (which is bad---we aren't allowed to
        # double-instantiate!)
        planner_exts = PlannerExtensions(
            service_config.pddl_files,
            service_config.init_problem_name,
            dg_use_act_history=self.use_history,
            dg_use_lm_cuts=self.use_lm_cuts,
            dg_heuristic_name=self.heuristic_data_gen_name)

        return InstantiatedNetwork(
            instantiator=self,
            policy=policy,
            planner_exts=planner_exts,
            single_problem=single_problem)
