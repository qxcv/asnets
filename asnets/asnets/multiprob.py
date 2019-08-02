"""Code for training on several problems at the same time. Their all live in
their own sandboxed Python interpreters so that they can have their own copies
of MDPSim and SSiPP."""

from copy import deepcopy
import ctypes
import getpass
import os
from multiprocessing import Process
import signal
import sys
from time import sleep, time
import uuid
import weakref

import rpyc
from rpyc.utils.server import OneShotServer

from asnets.prof_utils import try_save_profile
from asnets.py_utils import set_random_seeds


def parent_death_pact(signal=signal.SIGINT):
    """Commit to kill current process when parent process dies."""
    assert sys.platform == 'linux', \
        "this fn only works on Linux right now"
    libc = ctypes.CDLL("libc.so.6")
    # see include/uapi/linux/prctl.h in kernel
    PR_SET_PDEATHSIG = 1
    # last three args are unused for PR_SET_PDEATHSIG
    retcode = libc.prctl(PR_SET_PDEATHSIG, signal, 0, 0, 0)
    if retcode != 0:
        raise Exception("prctl() returned nonzero retcode %d" % retcode)


def start_server(service_args, socket_path):
    if service_args.random_seed is not None:
        set_random_seeds(service_args.random_seed)
    # avoid import cycle
    from asnets.supervised import make_problem_service
    parent_death_pact(signal=signal.SIGKILL)
    new_service = make_problem_service(service_args, set_proc_title=True)
    server = OneShotServer(new_service, socket_path=socket_path)
    print('Child process starting OneShotServer %s' % server)
    try:
        server.start()
    finally:
        # save kernprof profile for this subprocess if we can
        try_save_profile()


def to_local(obj):
    """Convert a NetRef to an object to something that's DEFINITELY local."""
    # can probably smarter here (e.g. not copying netrefs, using joblib for
    # efficient Numpy support); oh well
    # TODO: try using encode/decode with joblib instead! Could be much, much
    # faster.
    # TODO: make sure that you're transmitting observations as byte tensors
    # whenever possible (or at most float32s).
    return deepcopy(obj)


def wait_exists_polling(file_path, max_wait, *, delta=0.05):
    """Check if file exists every `delta` seconds. I'm using this to wait for a
    socket to get created by a subprocess."""
    start_time = time()
    while not os.path.exists(file_path):
        sleep(delta)
        if time() - start_time > max_wait:
            return False
    return True


class ProblemServer(object):
    """Spools up another process to host a ProblemService."""
    # how long we need to wait for the connection to spool up
    MAX_WAIT_TIME = 15.0

    def __init__(self, service_conf):
        # Sockets go in /tmp rather than cwd because Linux limits socket paths
        # (not filenames!) to 108 chars, and cwd might be too long (yes,
        # seriously!). The username is just in there to avoid case where
        # somebody else makes the dir & stops us from writing to it.
        user = getpass.getuser()
        sock_dir = f'/tmp/asnet-sockets-{user}/'
        os.makedirs(sock_dir, exist_ok=True)
        self._unix_sock_path = os.path.join(sock_dir,
                                            'socket.' + uuid.uuid4().hex)
        self._serve_proc = Process(
            target=start_server, args=(
                service_conf,
                self._unix_sock_path,
            ))
        self._serve_proc.start()
        self._start_time = time()

        self._conn = None

        # this ensures that we always close connection (& thus terminate server
        # on other end) before shutting down, no matter what
        # (basically weakref.finalize(obj, func) ensures that func is called
        # when obj is destroyed---presumably just beforehand)
        self._finalizer = weakref.finalize(self._serve_proc, self._kill_conn)

    def _kill_conn(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def stop(self):
        self._kill_conn()

        try:
            os.unlink(self._unix_sock_path)
        except FileNotFoundError:
            pass

        if self._serve_proc is not None:
            self._serve_proc.terminate()
            try:
                self._serve_proc.join(5)
            except Exception:
                print('Process is being difficult.')
                pid = self._serve_proc.pid
                if pid is not None and self._serve_proc.is_alive():
                    print('I know how to handle difficult processes.')
                    os.kill(pid, signal.SIGKILL)
                    self._serve_proc.join(5)
            self._serve_proc = None

    def __del__(self):
        if hasattr(self, '_serve_proc') and self._serve_proc is not None:
            print('Cleaning up server process in destructor')
            self.stop()

    def _get_rpyc_conn(self):
        if self._conn is None:
            to_wait = max(0, self.MAX_WAIT_TIME - (time() - self._start_time))
            if to_wait > 0:
                # It actually takes a few seconds for the background worker to
                # spool up and start accepting connections. Obviously it could
                # be more than self.MAX_WAIT_TIME, but I don't really have a
                # better way of doing things than this (mostly because all the
                # socket binding in RPyC happens in a monolithic "run
                # everything" method which I can't break up).
                print('Waiting at most %.2fs for rpyc connection' % to_wait)
                # ignore return value; we'll get an error later if the file
                # doesn't exist
                has_sock = wait_exists_polling(
                    self._unix_sock_path, max_wait=to_wait)
                print(f"Wait time up, got has_sock={has_sock}")
            sleep_time = 1.0
            print(f"Sleeping an extra {sleep_time}s to make sure conn is up")
            sleep(sleep_time)
            self._conn = rpyc.utils.factory.unix_connect(
                path=self._unix_sock_path)
            # we can unlink socket after connecting
            os.unlink(self._unix_sock_path)
        return self._conn

    @property
    def conn(self):
        return self._get_rpyc_conn()

    @property
    def service(self):
        # return handle on root service for connection, which in this case is a
        # ProblemService
        return self.conn.root
