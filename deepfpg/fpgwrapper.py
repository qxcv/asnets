#!/usr/bin/env python3
"""Used to coordinate MDPSim and FPG"""

from contextlib import contextmanager
from os import makedirs
from subprocess import run, PIPE, Popen, TimeoutExpired
from time import sleep
from xml.etree import ElementTree as ET


def run_fpgclient(fpgclient_path, problem, time_limit, port, ppddl_paths):
    """Connect to mdpsim with fpgclient and plan! Returns simulation results:
    num successes, num trials, average makespan for successes."""
    args = [
        fpgclient_path, '-N', problem, '-H', 'localhost', '-P', str(port),
        '-t', str(time_limit)
    ] + ppddl_paths
    proc = run(args, stdout=PIPE, check=True)

    lines = proc.stdout.decode('utf8').splitlines()
    end_lines = [l for l in lines if l.startswith('<end-session>')]
    tree = ET.fromstring(end_lines[-1])

    def ftext(s):
        return ' '.join(tree.find(s).itertext())

    num_successes = int(ftext('goals/reached/successes'))
    num_trials = int(ftext('rounds'))
    if num_successes > 0:
        mean_makespan = float(ftext('goals/reached/turn-average'))
    else:
        mean_makespan = None

    # here's the sort of thing we're looking for (in reality, all on one line
    # w/ no whitespace):
    #
    # <end-session>
    #   <sessionID>14</sessionID>
    #   <rounds>100</rounds>
    #   <goals>
    #     <failed>14</failed>
    #     <reached>
    #       <successes>86</successes>
    #       <time-average>8.63953</time-average>
    #       <turn-average>18.1395</turn-average>
    #     </reached>
    #   </goals>
    #   <metric-average>0.86</metric-average>
    # </end-session>

    return num_successes, num_trials, mean_makespan


@contextmanager
def mdpsim_ctx(ppddl_paths,
               mdpsim_path,
               port,
               round_limit=100,
               turn_limit=1000):
    # Give mdpsim some time to get set up
    print('Starting mdpsim')
    log_path = 'logs/'
    try:
        makedirs(log_path)
    except FileExistsError:
        pass
    args = [
        mdpsim_path, '-p', '-l', 'logs/', '-L', '512', '-R', str(round_limit),
        '-L', str(turn_limit), '-P', str(port)
    ] + ppddl_paths
    proc = Popen(args)
    sleep(0.5)

    # Make sure it's actually running
    if proc.poll() is None:
        print('mdpsim listening on port %d' % port)
    else:
        raise Exception('mdpsim exited early; something is wrong')

    try:
        yield
    finally:
        # Clean up once client is done with mdpsim
        print('Terminating mdpsim')
        proc.terminate()
        try:
            proc.wait(1)
        except TimeoutExpired:
            print("mdpsim stepped out of line and now it\'s kill -9")
            proc.kill()
