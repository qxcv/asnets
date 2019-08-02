"""Utilities for profiling code, based on kernprof."""

import builtins
import os
try:
    import kernprof
except ImportError:
    kernprof = None


def _has_profile():
    """Check whether we have kernprof & kernprof has given us global 'profile'
    object."""
    return kernprof is not None and hasattr(builtins, 'profile')


def can_profile(func):
    """Decorator which adds kernprof profiling to given function, if it's
    available."""
    if _has_profile():
        return builtins.profile(func)
    return func


def _kernprof_options():
    """Gets command line options given to kernprof."""
    # this is almost stupid enough to work.
    # explainer: kernprof 2.0 uses execfile(<run_asnets.py path>, locals(),
    # locals()) to run scripts. That means that run_asnets.py gets access to
    # all the internal stuff that kernprof uses to set up scripts---including
    # the 'options' object containing all command line options! We save those
    # globals in run_asnets.py so we can access them later :-)
    return _run_asnets_globals['options']  # noqa


def try_save_profile():
    """If there's a profiler, this tries to save a profile with appropriate
    filename. Relies on arguments being passed correctly."""
    # avoids flake8 warnings
    if _has_profile():
        options = _kernprof_options()
        pid = os.getpid()
        if options.outfile is not None:
            # append PID to destination name and save that
            real_dest = options.outfile + '.%d' % pid
            print("Subprocess %d saving stats to '%s'" % (pid, real_dest))
            builtins.profile.dump_stats(real_dest)
        if options.view is not None:
            print("Profiler stats for subprocess %d:" % pid)
            builtins.profile.print_stats()
