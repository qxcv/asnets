#!/usr/bin/env python3
"""Run an experiment using the standard Python-based configuration format (see
`experiments/` subdirectory for example.)"""

import argparse
import datetime
from hashlib import md5
from importlib import import_module
from os import path, makedirs, listdir, getcwd
from shutil import copytree
from subprocess import Popen, PIPE, TimeoutExpired
import sys
from time import time

import ray

THIS_DIR = path.dirname(path.abspath(__file__))
PLANNER_ROOT = path.abspath(path.join(THIS_DIR, '..', '..'))
# hack to ensure we can find 'experiments' module
sys.path.append(PLANNER_ROOT)


def extract_by_prefix(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):]


def get_pin_list():
    """Get list of CPU IDs to pin to, using Ray's CPU allocation."""
    resources = ray.get_resource_ids()
    cpu_ids = []
    for cpu_id, cpu_frac in resources['CPU']:
        # sanity check: we should have 100% of each CPU
        assert abs(cpu_frac - 1.0) < 1e-5, \
            "for some reason I have fraction %f of CPU %d (??)" \
            % (cpu_id, cpu_frac)
        cpu_ids.append(cpu_id)
    assert len(cpu_ids) > 0, \
        "Ray returned no CPU IDs (was num_cpus=0 accidentally specified " \
        "for this task?)"
    return cpu_ids


def run_asnets_local(flags, root_dir, need_snapshot, timeout, is_train,
                     enforce_ncpus, cwd):
    """Run ASNets code on current node. May be useful to wrap this in a
    ray.remote()."""
    cmdline = []
    if enforce_ncpus:
        pin_list = get_pin_list()
        pin_list_str = ','.join(map(str, pin_list))
        ts_cmd = ['taskset', '--cpu-list', pin_list_str]
        print('Pinning job with "%s"' % ' '.join(ts_cmd))
        cmdline.extend(ts_cmd)
    cmdline.extend(['python', '-m', 'asnets.scripts.run_asnets'] + flags)
    print('Running command line "%s"' % ' '.join(cmdline))

    # we use this for logging
    unique_suffix = md5(' '.join(cmdline).encode('utf8')).hexdigest()
    dest_dir = path.join(root_dir, 'runs', unique_suffix)
    print('Will write results to %s' % dest_dir)
    makedirs(dest_dir, exist_ok=True)
    with open(path.join(dest_dir, 'cmdline'), 'w') as fp:
        fp.write(' '.join(cmdline))
    stdout_path = path.join(dest_dir, 'stdout')
    stderr_path = path.join(dest_dir, 'stderr')

    dfpg_proc = tee_out_proc = tee_err_proc = None
    start_time = time()
    try:
        # print to stdout/stderr *and* save as well
        dfpg_proc = Popen(cmdline, stdout=PIPE, stderr=PIPE, cwd=cwd)
        # first tee for stdout
        tee_out_proc = Popen(['tee', stdout_path], stdin=dfpg_proc.stdout)
        # second tee for stderr
        tee_err_proc = Popen(['tee', stderr_path], stdin=dfpg_proc.stderr)

        # close descriptors from this proc (they confuse child error handling);
        # see https://stackoverflow.com/q/23074705
        dfpg_proc.stdout.close()
        dfpg_proc.stderr.close()

        # twiddle, twiddle, twiddle
        timed_out = False
        bad_retcode = False
        try:
            dfpg_proc.wait(timeout=timeout)
        except TimeoutExpired:
            # uh, oops; better kill everything
            print('Run timed out after %ss!' % timeout)
            timed_out = True
    finally:
        # "cleanup"
        for proc in [tee_out_proc, tee_err_proc, dfpg_proc]:
            if proc is None:
                continue
            proc.poll()
            if proc.returncode is None:
                # make sure it's dead
                print('Force-killing a process')
                proc.terminate()
            proc.wait()
            retcode = proc.returncode
            if retcode != 0:
                print('Process exited with code %s: %s' %
                      (retcode, ' '.join(proc.args)))
                bad_retcode = True

    # write out extra info
    elapsed_time = time() - start_time
    with open(path.join(dest_dir, 'elapsed_secs'), 'w') as fp:
        fp.write('%f\n' % elapsed_time)
    with open(path.join(dest_dir, 'termination_status'), 'w') as fp:
        fp.write('timed_out: %s\nbad_retcode: %s\n' % (timed_out, bad_retcode))
    if is_train:
        with open(path.join(dest_dir, 'is_train'), 'w') as fp:
            # presence of 'is_train' file (in this case containing just a
            # newline) is sufficient to indicate that this was a train run
            print('', file=fp)

    # get stdout for... reasons
    with open(stdout_path, 'r') as fp:
        stdout = fp.read()
    lines = stdout.splitlines()

    # copy all info in custom dir into original prog's output dir (makes it
    # easier to associated)
    run_subdir = extract_by_prefix(lines, 'Unique prefix: ')
    if run_subdir is None:
        raise Exception("Couldn't find unique prefix for problem!")
    run_dir = path.join(root_dir, run_subdir)
    copytree(dest_dir, path.join(run_dir, 'run-info'))

    if need_snapshot:
        # parse output to figure out where it put the last checkpoint
        final_checkpoint_dir = extract_by_prefix(lines, 'Snapshot directory: ')
        if final_checkpoint_dir is None:
            msg = "cannot find final snapshot from stdout; check logs!"
            raise Exception(msg)
        # choose latest snapshot
        by_num = {}
        snaps = [
            path.join(final_checkpoint_dir, bn)
            for bn in listdir(final_checkpoint_dir)
            if bn.startswith('snapshot_')
        ]
        for snap in snaps:
            bn = path.basename(snap)
            num_s = bn.split('_')[1].rsplit('.', 1)[0]
            if num_s == 'final':
                # always choose this
                num = float('inf')
            else:
                num = int(num_s)
            by_num[num] = snap
        if len(by_num) == 0:
            msg = "could not find any snapshots in '%s'" % final_checkpoint_dir
            raise Exception(msg)
        # if this fails then we don't have any snapshots
        final_checkpoint_path = by_num[max(by_num.keys())]

        return final_checkpoint_path


def build_arch_flags(arch_mod, is_train):
    """Build flags which control model arch and training strategy."""
    flags = []
    assert arch_mod.SUPERVISED, "only supervised training supported atm"
    if is_train:
        flags.extend(['--dropout', str(arch_mod.DROPOUT)])
    if not arch_mod.SKIP:
        flags.append('--no-skip')
    if arch_mod.DET_EVAL:
        flags.append('--det-eval')
    if not arch_mod.USE_LMCUT_FEATURES:
        flags.append('--no-use-lm-cuts')
    if arch_mod.USE_ACT_HISTORY_FEATURES:
        flags.append('--use-act-history')
    if arch_mod.TEACHER_EXPERIENCE_MODE == 'ROLLOUT':
        flags.append('--no-use-teacher-envelope')
    elif arch_mod.TEACHER_EXPERIENCE_MODE != 'ENVELOPE':
        raise ValueError(
            f"Unknown experience mode '{arch_mod.TEACHER_EXPERIENCE_MODE}'; "
            "try 'ROLLOUT' or 'ENVELOPE'")
    if arch_mod.L1_REG:
        assert isinstance(arch_mod.L1_REG, (float, int))
        l1_reg = str(arch_mod.L1_REG)
    else:
        l1_reg = '0.0'
    if arch_mod.L2_REG:
        assert isinstance(arch_mod.L2_REG, (float, int))
        l2_reg = str(arch_mod.L2_REG)
    else:
        l2_reg = '0.0'
    flags.extend([
        '--num-layers', str(arch_mod.NUM_LAYERS),
        '--hidden-size', str(arch_mod.HIDDEN_SIZE),
        '--target-rollouts-per-epoch', str(arch_mod.TARGET_ROLLOUTS_PER_EPOCH),
        '--l2-reg', l2_reg,
        '--l1-reg', l1_reg,
        '-R', str(arch_mod.EVAL_ROUNDS),
        '-L', str(arch_mod.ROUND_TURN_LIMIT),
        '-t', str(arch_mod.TIME_LIMIT_SECONDS),
        '--supervised-lr', str(arch_mod.SUPERVISED_LEARNING_RATE),
        '--supervised-bs', str(arch_mod.SUPERVISED_BATCH_SIZE),
        '--supervised-early-stop', str(arch_mod.SUPERVISED_EARLY_STOP),
        '--save-every', str(arch_mod.SAVE_EVERY_N_EPOCHS),
        '--ssipp-teacher-heur', arch_mod.SSIPP_TEACHER_HEURISTIC,
        '--opt-batch-per-epoch', str(arch_mod.OPT_BATCH_PER_EPOCH),
        '--teacher-planner', arch_mod.TEACHER_PLANNER,
        '--sup-objective', arch_mod.TRAINING_STRATEGY,
    ])  # yapf: disable
    if arch_mod.LEARNING_RATE_STEPS:
        for k, r in arch_mod.LEARNING_RATE_STEPS:
            assert k > 0, r > 0
            assert isinstance(k, int)
            assert isinstance(k, (int, float))
            flags.extend(['--lr-step', str(k), str(r)])
    return flags


def add_prefix(prefix, filenames):
    """Add a prefix directory to a bunch of filenames."""
    return [path.join(prefix, fn) for fn in filenames]


def build_prob_flags_train(prob_mod):
    """Build up some train flags for ASNets."""
    pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.COMMON_PDDLS)
    train_pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.TRAIN_PDDLS)
    pddls.extend(train_pddls)
    other_flags = []
    if prob_mod.TRAIN_NAMES:
        for tn in prob_mod.TRAIN_NAMES:
            other_flags.extend(['-p', tn])
    return other_flags + pddls


def build_prob_flags_test(prob_mod, allowed_idxs=None):
    """Build a list of flag sets, with one flag set for each requested
    experiment."""
    pddls = add_prefix(prob_mod.PDDL_DIR, prob_mod.COMMON_PDDLS)
    rv = []
    for idx, path_and_name in enumerate(prob_mod.TEST_RUNS):
        pddl_paths, prob_name = path_and_name
        if allowed_idxs is not None and idx not in allowed_idxs:
            print('Will skip item %d: %s' % (idx, path_and_name))
            continue
        prob_flag = []
        if prob_name is not None:
            prob_flag = ['-p', prob_name]
        these_pddls = add_prefix(prob_mod.PDDL_DIR, pddl_paths)
        rv.append((idx, prob_flag + pddls + these_pddls))
    return rv


def get_prefix_dir(checkpoint_path):
    """Turn path like experiments-results/experiments.actprop_2l-.../.../... into
    experiment-results/experiments.actprop_2l.../"""
    real_path = path.abspath(checkpoint_path)
    parts = real_path.split(path.sep)
    for idx in range(len(parts) - 1)[::-1]:
        part = parts[idx]
        if part.startswith('experiments.'):
            return path.sep.join(parts[:idx + 1])
    raise ValueError("Couldn't find experiments. prefix in '%s'" %
                     checkpoint_path)


def parse_idx_list(idx_list):
    idx_strs = [int(s) for s in idx_list.split(',') if s.strip()]
    return idx_strs


parser = argparse.ArgumentParser(description='Run an experiment with ASNets')
parser.add_argument('--resume-from',
                    default=None,
                    help='resume experiment from given checkpoint path')
parser.add_argument(
    '--restrict-test-probs',
    default=None,
    type=parse_idx_list,
    help='takes comma-separated list of evaluation problem numbers to test')
parser.add_argument(
    '--job-ncpus',
    type=int,
    default=None,
    help='number of CPUs *per job* (must be <= --ray-ncpus; default is 1)')
parser.add_argument(
    '--enforce-job-ncpus',
    default=False,
    action='store_true',
    help='enforce --job-ncpus usage by using taskset/sched_setaffinity to '
    'pin jobs to unique cores')
parser.add_argument(
    '--ray-connect',
    default=None,
    help='connect Ray to this Redis DB instead of starting new cluster')
parser.add_argument(
    '--ray-ncpus',
    default=None,
    type=int,
    help='restrict Ray pool to use this many CPUs *in total* (only valid if '
    'spinning up new Ray cluster)')
parser.add_argument(
    'arch_module',
    metavar='arch-module',
    help='import path for Python file with architecture config (e.g. '
    '"experiments.actprop_1l")')
parser.add_argument(
    'prob_module',
    metavar='prob-module',
    help='import path for Python file with problem config (e.g. '
    '"experiments.ex_blocksworld")')


def main():
    args = parser.parse_args()

    # 1. load config
    print('Importing architecture from %s' % args.arch_module)
    arch_mod = import_module(args.arch_module)
    print('Importing problem from %s' % args.prob_module)
    prob_mod = import_module(args.prob_module)

    # 2. spool up Ray
    new_cluster = args.ray_connect is None
    ray_kwargs = {}
    if not new_cluster:
        ray_kwargs["redis_address"] = args.ray_connect
        assert args.ray_ncpus is None, \
            "can't provide --ray-ncpus and --ray-connect"
    else:
        if args.ray_ncpus is not None:
            assert args.job_ncpus is None \
                    or args.job_ncpus <= args.ray_ncpus, \
                    "must have --job-ncpus <= --ray-ncpus if both given"
            ray_kwargs["num_cpus"] = args.ray_ncpus
    ray.init(**ray_kwargs)
    main_inner(arch_mod=arch_mod,
               prob_mod=prob_mod,
               job_ncpus=args.job_ncpus,
               resume_from=args.resume_from,
               enforce_job_ncpus=args.enforce_job_ncpus)
    print('Fin :-)')


def main_inner(*,
               arch_mod,
               prob_mod,
               job_ncpus,
               enforce_job_ncpus,
               resume_from=None,
               restrict_test_probs=None):
    run_asnets_ray = ray.remote(num_cpus=job_ncpus)(run_asnets_local)
    root_cwd = getcwd()

    arch_name = arch_mod.__name__
    prob_name = prob_mod.__name__
    if resume_from is None:
        time_str = datetime.datetime.now().isoformat()
        prefix_dir = 'experiment-results/%s-%s-%s' % (prob_name, arch_name,
                                                      time_str)
        prefix_dir = path.join(root_cwd, prefix_dir)
        print('Will put everything in %s' % prefix_dir)

        # 3. train network
        print('\n\n\n\n\n\nTraining network')
        train_flags = [
            # log and snapshot dirs
            '-e', prefix_dir,
        ]  # yapf: disable
        train_flags.extend(build_arch_flags(arch_mod, is_train=True))
        train_flags.extend(build_prob_flags_train(prob_mod))
        final_checkpoint = ray.get(
            run_asnets_ray.remote(
                flags=train_flags,
                # we make sure it runs cmd in same dir as us,
                # because otherwise Ray subprocs freak out
                cwd=root_cwd,
                root_dir=prefix_dir,
                need_snapshot=True,
                is_train=True,
                enforce_ncpus=enforce_job_ncpus,
                timeout=arch_mod.TIME_LIMIT_SECONDS))
        print('Last valid checkpoint is %s' % final_checkpoint)
    else:
        final_checkpoint = resume_from
        prefix_dir = get_prefix_dir(final_checkpoint)
        print('Resuming from checkpoint "%s"' % final_checkpoint)
        print('Using experiment dir "%s"' % prefix_dir)

    # 4. test network
    print('\n\n\n\n\n\nTesting network')
    main_test_flags = [
        '--no-train',
        # avoid writing extra snapshot & TB files
        '--minimal-file-saves',
        '--resume-from', final_checkpoint,
        '-e', prefix_dir,
    ]  # yapf: disable
    main_test_flags.extend(build_arch_flags(arch_mod, is_train=False))

    print('Starting test loop')
    prob_flag_list = build_prob_flags_test(prob_mod, restrict_test_probs)
    job_infos = {}
    for prob_idx, test_prob_flags in prob_flag_list:
        print('Launching test on problem %d' % (prob_idx + 1))
        full_flags = main_test_flags + test_prob_flags
        job = run_asnets_ray.remote(
            flags=full_flags,
            root_dir=prefix_dir,
            cwd=root_cwd,
            need_snapshot=False,
            is_train=False,
            enforce_ncpus=enforce_job_ncpus,
            # run_asnets.py has its own timeout which it should obey, so give
            # it some slack
            timeout=arch_mod.EVAL_TIME_LIMIT_SECONDS + 30)
        job_infos[job] = (prob_idx, test_prob_flags)

    print("Waiting for jobs to finish")
    remaining = list(job_infos)
    while remaining:
        (ready, ), remaining = ray.wait(remaining, num_returns=1)
        prob_idx, test_prob_flags = job_infos[ready]
        print("Finished job %d (flags: %s)" % (prob_idx, test_prob_flags))

    # return the prefix_dir because hype.py needs that to figure out where to
    # point collate_results at
    return prefix_dir


if __name__ == '__main__':
    main()
