"""Stop future private job subprocess groups before the event packaging window."""
from datetime import datetime,timezone
import os
import signal
import subprocess
import time

OFFICIAL_FREEZE_UTC='2026-10-15T07:30:00+00:00'
JOB_STOP_UTC='2026-10-15T06:00:00+00:00' # 90 minutes for download, review, packaging and push.
JOB_STOP_TIMESTAMP=datetime.fromisoformat(JOB_STOP_UTC).timestamp()


def run_before_deadline(command,maximum_seconds,*,stop_timestamp=JOB_STOP_TIMESTAMP):
    """Use a new process group so timeout also stops process-pool descendants."""
    remaining=stop_timestamp-time.time()
    # Reserve kill grace inside the absolute cutoff. A late queued notebook
    # never obtains a new six-hour allowance extending beyond the event freeze.
    if remaining<=5:
        raise TimeoutError('Bluff preparation cutoff reached; preserve the verified incumbent.')
    limit=min(float(maximum_seconds),remaining-5)
    if limit<=0:raise ValueError('positive per-process time budget required')
    process=subprocess.Popen(command,start_new_session=True)
    try:
        code=process.wait(timeout=limit)
        if code:raise subprocess.CalledProcessError(code,command)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
        # The leader can exit before a descendant; kill the entire original
        # group even when the immediate child's return code is already known.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
        raise
    return subprocess.CompletedProcess(command,code)
