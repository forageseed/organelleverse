"""Native Windows process-tree cleanup; all code comes from the formal OV tree."""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Win32 Job Object contract")


def test_request_handshake_job_closes_worker_and_its_descendant(tmp_path):
    import psutil
    from organelleverse.capabilities._windows_job import WindowsWorkerJob

    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess,sys,time\nsys.stdin.readline()\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "print(child.pid,flush=True)\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [getattr(sys, "_base_executable", sys.executable), "-I", "-S", str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = WindowsWorkerJob(int(process._handle))
    try:
        process.stdin.write("start\n")
        process.stdin.flush()
        child_pid = int(process.stdout.readline())
        assert psutil.pid_exists(child_pid)
        job.close()
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not psutil.pid_exists(child_pid)
    finally:
        job.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_unassignable_handle_fails_without_a_permissive_fallback():
    from organelleverse.capabilities._windows_job import WindowsWorkerJob

    with pytest.raises(OSError):
        WindowsWorkerJob(0)
