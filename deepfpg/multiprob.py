"""Code for training on several problems at the same time. Their all live in
their own sandboxed Python interpreters so that they can have their own copies
of MDPSim and SSiPP."""

from copy import deepcopy
from multiprocessing import Process
from socket import socket, AF_INET, SOCK_STREAM
from time import sleep, time
import os
import signal

import rpyc
from rpyc.utils.server import OneShotServer

from prof_utils import try_save_profile


def open_port_number():
    """Create a socket to get an open ephemeral port number. That number can
    then be passed to RPyC to create a new sandbox process. Obviously this is
    vulnerable to a race condition, but it's easier than forking RPyC and
    fixing it so that rpyc.utils.server.Server.start() actually lets you grab a
    bound port number."""
    # See https://stackoverflow.com/a/2838309
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(('localhost', 0))
    _, pno = sock.getsockname()
    sock.close()
    return pno


def start_server(service_args, port):
    # avoid import cycle
    from supervised import make_problem_service
    new_service = make_problem_service(service_args)
    server = OneShotServer(new_service, hostname='localhost', port=port)
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
    # TODO: try using encode/decode with joblib instead! Should be much, much
    # faster.
    # TODO: make sure that you're transmitting observations as byte tensors
    # whenever possible (or at most float32s).
    return deepcopy(obj)


class ProblemServer(object):
    """Spools up another process to host a ProblemService."""
    # how long we need to wait for the connection to spool up
    WAIT_TIME = 3

    def __init__(self, service_conf):
        # we use a pipe to get a bound port number (it's ephemeral) from the
        # child
        self._port = open_port_number()
        self._serve_proc = Process(
            target=start_server, args=(service_conf, self._port, ))
        self._serve_proc.start()
        self._start_time = time()

        self._conn = None

    def stop(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._serve_proc is not None:
            self._serve_proc.terminate()
            try:
                self._serve_proc.join(5)
            except:
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
            to_wait = max(0, self.WAIT_TIME - (time() - self._start_time))
            if to_wait > 0:
                # It actually takes a few seconds for the background worker to
                # spool up and start accepting connections. Obviously it could
                # be more than self.WAIT_TIME, but I don't really have a better
                # way of doing things than this (mostly because all the socket
                # binding in RPyC happens in a monolithic "run everything"
                # method which I can't break up).
                print('Waiting %.2fs for rpyc connection' % to_wait)
                sleep(to_wait)
            self._conn = rpyc.connect(host='localhost', port=self._port)
        return self._conn

    @property
    def conn(self):
        return self._get_rpyc_conn()

    @property
    def service(self):
        # return handle on root service for connection, which in this case is a
        # ProblemService
        return self.conn.root
