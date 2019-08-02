#!/usr/bin/env python3

from importlib import import_module
import os
import os.path as osp
import random

import click
import ray
import tqdm

from asnets.fd_interface import run_fd_raw, STDOUT_BN
from asnets.pddl_utils import extract_domain_problem

# give us 3h to match the 2h30m training + X minutes execution time of ASNets
# (I think ASNets can take up to 1h on some problems, but are usually closer to
# like 10m)
TIMEOUT_MIN = 3 * 60
MAX_MEMORY_GB = 16  # double IPC; this is actually quite a bit of RAM
BEST_COST_PREFIX = "Best solution cost so far: "
THIS_DIR = osp.dirname(osp.abspath(__file__))
ASNETS_ROOT = osp.abspath(osp.join(THIS_DIR, '../../'))
RESULTS_DIR = osp.join(ASNETS_ROOT, 'experiment-results/baselines-det/')
PLANNERS = [
    'gbf-lmcut',  # does LM-cut solve this problem on its own?
    'astar-lmcut',  # does LM-cut help *at all*?
    'astar-lmcount',  # non-adm heuristic that I'm using as a det teacher
    'lama-2011',
    'lama-first'  # actual good planners
]
# cost bound for plans (this gets passed to search routines to cut off search
# at a certain depth)
MAX_ACTIONS = 300


@ray.remote(num_cpus=1)
def do_fd_run(planner, domain, problem):
    cwd = os.getcwd()
    domain_path = osp.join(cwd, domain)
    problem_path = osp.join(cwd, problem)
    _, domain_name, _, problem_name = extract_domain_problem(
        [domain_path, problem_path])
    dname = '%s:%s:%s' % (planner, domain_name, problem_name)
    result_dir = osp.join(RESULTS_DIR, dname)
    with open(domain_path) as dom_fp, open(problem_path) as prob_fp:
        domain_txt = dom_fp.read()
        prob_txt = prob_fp.read()
    subproc_rv = run_fd_raw(planner=planner,
                            domain_txt=domain_txt,
                            problem_txt=prob_txt,
                            result_dir=result_dir,
                            timeout_s=int(TIMEOUT_MIN * 60),
                            mem_limit_mb=int(MAX_MEMORY_GB * 1024),
                            cost_bound=MAX_ACTIONS)
    # could use rv.returncode to check success, but that is sometimes nonzero
    # even when the planner manages to find a rough plan :(
    try:
        with open(osp.join(result_dir, STDOUT_BN), 'r') as out_fp:
            lines = out_fp.read().splitlines()
            success = any(l for l in lines if l.startswith(BEST_COST_PREFIX)) \
                or subproc_rv.returncode == 0
    except IOError:
        success = False
    return (planner, domain, problem), success


def get_module_test_problems(prob_mod_name):
    """Get the path to the domain and a list of paths to problems from a given
    experiment module name (e.g "experiments.det_gripper")."""
    prob_mod = import_module(prob_mod_name)
    domain_fname, = prob_mod.COMMON_PDDLS
    domain_path = osp.join(ASNETS_ROOT, prob_mod.PDDL_DIR, domain_fname)
    problems_names = []
    for prob_fnames, name in prob_mod.TEST_RUNS:
        prob_fname, = prob_fnames
        prob_path = osp.join(ASNETS_ROOT, prob_mod.PDDL_DIR, prob_fname)
        # this is hashable! Woo!
        problems_names.append((prob_path, name))
    return domain_path, problems_names


def wait_all_unordered(tasks):
    # wait on all tasks & return results in arbitrary order
    rv = []
    remaining = tasks
    print("Waiting for %d tasks to complete" % len(tasks))
    tq = tqdm.trange(len(tasks))
    while remaining:
        (done, ), remaining = ray.wait(remaining)
        rv.append(ray.get(done))
        tq.update(1)
    tq.close()
    print("Tasks complete")
    return rv


def ray_setup(connect, ncpus):
    print("Setting up Ray")
    new_cluster = connect is None
    ray_kwargs = {}
    if not new_cluster:
        ray_kwargs["redis_address"] = connect
        assert ncpus is None, \
            "can't provide --ray-ncpus and --ray-connect"
    elif ncpus is not None:
        ray_kwargs['num_cpus'] = ncpus
    ray.init(**ray_kwargs)


@click.command(help='run deterministic baseline planners')
@click.argument('problem_module')
@click.option(
    '--ray-connect',
    default=None,
    type=str,
    help='connect Ray to this Redis DB instead of starting new cluster')
@click.option(
    '--ray-ncpus',
    default=None,
    type=int,
    help='restrict Ray pool to use this many CPUs *in total* (only valid if '
    'spinning up new Ray cluster)')
def main(problem_module, ray_connect, ray_ncpus):
    # go back to root dir (in case running from script)
    os.chdir(ASNETS_ROOT)

    # load up all problems
    print('Importing problem from %s' % problem_module)
    domain_path, problems_names = get_module_test_problems(problem_module)
    problems = []
    for problem, name in problems_names:
        assert name is None, "I don't support named problems yet (should " \
            "be easy to do though)"
        problems.append(problem)
    planners_domains_problems = [(planner, domain_path, problem_path)
                                 for planner in PLANNERS
                                 for problem_path in problems]
    # some rudimentary load balancing
    random.shuffle(planners_domains_problems)

    # spin up or connect to Ray cluster
    ray_setup(connect=ray_connect, ncpus=ray_ncpus)

    # now run all jobs!
    print("Spawning Ray tasks")
    tasks = [do_fd_run.remote(*pdp) for pdp in planners_domains_problems]
    wait_all_unordered(tasks)


if __name__ == '__main__':
    main()
