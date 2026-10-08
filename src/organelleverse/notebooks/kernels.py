"""Isolated Python and R kernel sessions over a length-prefixed protocol.

One Python session and one R session per coordinator; they are separate
processes sharing no memory. The wire format is deliberately trivial so both
language workers can implement it without extra packages: three length-
prefixed UTF-8 text frames per request (code, inputs manifest, outputs
manifest; manifest lines are ``name=path``) and one length-prefixed JSON
response frame ``{"ok": bool, "stdout": str, "result": str|null, "error":
str|null}``. The R worker builds its fixed-shape JSON response itself.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.core.external import MESSAGE_TAIL_LINES, STDERR_TAIL_LINES, tail_lines

__all__ = ["PYTHON_WORKER_SOURCE", "KernelResponse", "KernelSession"]

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

_FRAME = struct.Struct(">I")
_STARTUP_TIMEOUT = 15.0
_EXECUTE_TIMEOUT = 120.0


@dataclass(frozen=True)
class KernelResponse:
    ok: bool
    stdout: str
    result: str | None
    error: str | None


PYTHON_WORKER_SOURCE = r'''
import io, json, struct, sys, contextlib

namespace = {}
while True:
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        break
    (length,) = struct.unpack(">I", header)
    code = sys.stdin.buffer.read(length).decode("utf-8")
    (n_inputs,) = struct.unpack(">I", sys.stdin.buffer.read(4))
    inputs = {}
    for _ in range(n_inputs):
        (n,) = struct.unpack(">I", sys.stdin.buffer.read(4))
        name, _, path = sys.stdin.buffer.read(n).decode("utf-8").partition("=")
        inputs[name] = path
    (n_outputs,) = struct.unpack(">I", sys.stdin.buffer.read(4))
    outputs = {}
    for _ in range(n_outputs):
        (n,) = struct.unpack(">I", sys.stdin.buffer.read(4))
        name, _, path = sys.stdin.buffer.read(n).decode("utf-8").partition("=")
        outputs[name] = path
    namespace["INPUT_PATHS"] = inputs
    namespace["OUTPUT_PATHS"] = outputs
    buffer = io.StringIO()
    error = None
    result = None
    try:
        import ast
        tree = ast.parse(code, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = tree.body.pop()
            with contextlib.redirect_stdout(buffer):
                exec(compile(tree, "<cell>", "exec"), namespace)
                result = repr(eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace))
        else:
            with contextlib.redirect_stdout(buffer):
                exec(compile(tree, "<cell>", "exec"), namespace)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    payload = json.dumps(
        {"ok": error is None, "stdout": buffer.getvalue(), "result": result, "error": error}
    ).encode("utf-8")
    sys.stdout.buffer.write(struct.pack(">I", len(payload)) + payload)
    sys.stdout.buffer.flush()
'''


def _error(code: str, message: str, **details: object) -> OrganelleDependencyError:
    return OrganelleDependencyError(code=code, message=message, details=details)


class KernelSession:
    """One persistent worker process for one language."""

    def __init__(
        self,
        language: Literal["python", "r"],
        *,
        python_executable: str = "",
    ) -> None:
        self._language = language
        self._python = python_executable or sys.executable
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_sink: BinaryIO | None = None

    @property
    def language(self) -> Literal["python", "r"]:
        return self._language  # pyright: ignore[reportReturnType]

    def _stderr_tail(self) -> str:
        """Bounded tail of everything the kernel process wrote to stderr."""
        sink = self._stderr_sink
        if sink is None or sink.closed:
            return ""
        try:
            end = sink.seek(0, os.SEEK_END)
            sink.seek(max(0, end - 20_000))
            captured = sink.read()
        except OSError:
            return ""
        return tail_lines(captured, STDERR_TAIL_LINES)

    def _process_lost_error(
        self, message: str, *, returncode: int | None
    ) -> OrganelleDependencyError:
        tail = self._stderr_tail()
        if tail:
            message = f"{message}: {tail_lines(tail, MESSAGE_TAIL_LINES)}"
        return _error(
            "kernel.process_lost",
            message,
            language=self._language,
            returncode=returncode,
            stderr_tail=tail,
        )

    def _ensure_started(self) -> subprocess.Popen[bytes]:
        if self._process is not None:
            if self._process.poll() is None:
                return self._process
            raise self._process_lost_error(
                "this kernel session died; construct a new session for a fresh kernel",
                returncode=self._process.returncode,
            )
        if self._language == "python":
            command = [self._python, "-I", "-c", PYTHON_WORKER_SOURCE]
        else:
            rscript = shutil.which("Rscript")
            if rscript is None:
                raise _error(
                    "kernel.r_unavailable",
                    "no Rscript executable is available on PATH",
                )
            shim = Path(__file__).parent / "resources" / "kernel_shim.R"
            command = [rscript, "--vanilla", str(shim)]
        # stderr spools to a private temp file: a pipe could deadlock once
        # its buffer fills, and DEVNULL would discard the startup cause
        # (bad interpreter, R shim failure) the process_lost report surfaces.
        self._stderr_sink = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 (closed in shutdown)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_sink,
        )
        return self._process

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def execute(
        self,
        code: str,
        *,
        inputs: Mapping[str, str] = {},
        outputs: Mapping[str, str] = {},
        timeout: float = _EXECUTE_TIMEOUT,
    ) -> KernelResponse:
        process = self._ensure_started()
        assert process.stdin is not None and process.stdout is not None
        payload = code.encode("utf-8")
        process.stdin.write(_FRAME.pack(len(payload)) + payload)
        for manifest in (inputs, outputs):
            process.stdin.write(_FRAME.pack(len(manifest)))
            for name, path in manifest.items():
                line = f"{name}={path}".encode()
                process.stdin.write(_FRAME.pack(len(line)) + line)
        process.stdin.flush()
        header = process.stdout.read(4)
        if len(header) < 4:
            process.poll()
            raise self._process_lost_error(
                "the kernel process exited before responding",
                returncode=process.returncode,
            )
        (length,) = _FRAME.unpack(header)
        raw = json.loads(process.stdout.read(length).decode("utf-8"))
        if not raw.get("ok") and raw.get("error") is None:
            raise _error(
                "kernel.protocol_invalid",
                "the kernel returned a malformed response",
                language=self._language,
            )
        return KernelResponse(
            ok=bool(raw.get("ok")),
            stdout=str(raw.get("stdout") or ""),
            result=raw.get("result"),
            error=raw.get("error"),
        )

    def shutdown(self) -> None:
        process = self._process
        self._process = None
        sink = self._stderr_sink
        self._stderr_sink = None
        if sink is not None:
            sink.close()
        if process is None or process.poll() is not None:
            return
        assert process.stdin is not None
        try:
            process.stdin.close()
            process.wait(timeout=_STARTUP_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=_STARTUP_TIMEOUT)
