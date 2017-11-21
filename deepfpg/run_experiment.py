#!/usr/bin/env python3
"""Run an experiment using the standard Python-based configuration format (see
`experiments/` subdirectory for example.)"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
from os import path, makedirs, listdir
from time import time
from shutil import copytree
from subprocess import Popen, PIPE, TimeoutExpired
from hashlib import md5
from importlib import import_module


def extract_by_prefix(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix):]


def run_dfpg(flags, root_dir, need_snapshot=True, timeout=None):
    """Run new, 'deep' planner."""
    cmdline = ['./fpg.py'] + flags
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
        dfpg_proc = Popen(cmdline, stdout=PIPE, stderr=PIPE)
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
    flags.append('--supervised')
    if is_train:
        flags.extend(['-O', arch_mod.TRAIN_MODEL_FLAGS])
    else:
        flags.extend(['-O', arch_mod.TEST_MODEL_FLAGS])
    if arch_mod.DET_EVAL:
        flags.append('--det-eval')
    if not arch_mod.USE_LMCUT_FEATURES:
        flags.append('--no-use-lm-cuts')
    flags.extend([
        '-R', str(arch_mod.EVAL_ROUNDS),
        '-m', arch_mod.MODEL_TYPE,
        '-L', str(arch_mod.ROUND_TURN_LIMIT),
        '-t', str(arch_mod.TIME_LIMIT_SECONDS),
        '--supervised-lr', str(arch_mod.SUPERVISED_LEARNING_RATE),
        '--supervised-bs', str(arch_mod.SUPERVISED_BATCH_SIZE),
        '--supervised-teacher-heur', arch_mod.SUPERVISED_TEACHER_HEURISTIC
    ])  # yapf: disable
    return flags


def add_prefix(prefix, filenames):
    """Add a prefix directory to a bunch of filenames."""
    return [path.join(prefix, fn) for fn in filenames]


def build_prob_flags_train(prob_mod):
    """Build up some train flags for Deep FPG."""
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


parser = argparse.ArgumentParser(description='Run an experiment with DFPG')
parser.add_argument(
    '--resume-from',
    default=None,
    help='resume experiment from given checkpoint path')
parser.add_argument(
    '--restrict-test-probs',
    default=None,
    type=parse_idx_list,
    help='takes comma-separated list of evaluation problem numbers to test')
parser.add_argument(
    '--par-eval',
    type=int,
    default=1,
    help='number of environments to evaluate on at once')
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

    arch_name = arch_mod.__name__
    prob_name = prob_mod.__name__
    if args.resume_from is None:
        time_str = datetime.datetime.now().isoformat()
        prefix_dir = './experiment-results/%s-%s-%s' % (prob_name, arch_name,
                                                        time_str)
        print('Will put everything in %s' % prefix_dir)

        # 2. train network
        print('\n\n\n\n\n\nTraining network')
        train_flags = [
            # log and snapshot dirs
            '-e', prefix_dir,
        ]  # yapf: disable
        train_flags.extend(build_arch_flags(arch_mod, is_train=True))
        train_flags.extend(build_prob_flags_train(prob_mod))
        final_checkpoint = run_dfpg(
            train_flags,
            prefix_dir,
            need_snapshot=True,
            timeout=arch_mod.TIME_LIMIT_SECONDS)
        print('Last valid checkpoint is %s' % final_checkpoint)
    else:
        final_checkpoint = args.resume_from
        prefix_dir = get_prefix_dir(final_checkpoint)
        print('Resuming from checkpoint "%s"' % final_checkpoint)
        print('Using experiment dir "%s"' % prefix_dir)

    # 3. test network
    print('\n\n\n\n\n\nTesting network')
    main_test_flags = [
        '--no-train',
        '--resume-from', final_checkpoint,
        '-e', prefix_dir,
    ]  # yapf: disable
    main_test_flags.extend(build_arch_flags(arch_mod, is_train=False))

    # do several evals at once
    def runner(args):
        prob_idx, test_prob_flags = args
        print('Testing on test problem %d' % (prob_idx + 1))
        full_flags = main_test_flags + test_prob_flags
        return run_dfpg(
            full_flags,
            prefix_dir,
            need_snapshot=False,
            # fpg.py has its own timeout which it should obey, so give it some
            # slack
            timeout=arch_mod.TIME_LIMIT_SECONDS + 30)

    print('Initiating test pool of %d threads and running DFPG' %
          args.par_eval)
    prob_flag_list = build_prob_flags_test(prob_mod, args.restrict_test_probs)
    with ThreadPoolExecutor(max_workers=args.par_eval) as executor:
        for result in executor.map(runner, prob_flag_list):
            # don't care, just want to evaluate all
            pass

    print('Fin :-)')


if __name__ == '__main__':
    main()
