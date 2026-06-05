"""
Exclusive file-lock helper shared by all CLI entry points.

Usage::

    from photovault_categorizer.lock import with_lock

    with with_lock():
        run_job()

If the lock file is already held by another process, ``with_lock()`` exits the
process with code 0 (not an error — the caller's systemd timer should succeed).
"""

import fcntl
import logging
import sys
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

LOCK_PATH = "/tmp/photovault-categorize.lock"


@contextmanager
def with_lock(lock_path: str = LOCK_PATH) -> Generator[None, None, None]:
    """Context manager that acquires an exclusive non-blocking file lock.

    Exits the process with code 0 if the lock is already held so systemd
    timers do not report the overlap as a failure.
    """
    lock_file = open(lock_path, "w")  # noqa: WPS515
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info(
            "Another categorizer run is active (lock held at %s) — exiting 0",
            lock_path,
        )
        lock_file.close()
        sys.exit(0)

    try:
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
