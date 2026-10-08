"""Lightweight, importable evaluator fixtures for forkserver tests."""

from __future__ import annotations

import multiprocessing
import subprocess
import sys
import time
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import cast

from organelleverse.optimization.strategy_models import EvaluatorBinding
from organelleverse.optimization.study_models import EvaluationRequest
from organelleverse.plugin_experiments.adaptive_models import RepeatEvidence

ACCURACY = "/metrics/optimization_evaluation/accuracy"
MEMORY = "/metrics/optimization_evaluation/peak_memory"
MEMORY_LIMIT = 256 * 1024 * 1024
_SHARED_MANAGER: SyncManager | None = None


def _shared_manager() -> SyncManager:
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = multiprocessing.get_context("spawn").Manager()
    return _SHARED_MANAGER


class FakeEvaluatorRunner:
    """Typed deterministic evaluator with a cross-process invocation count."""

    def __init__(
        self,
        binding: EvaluatorBinding,
        *,
        improve: bool = True,
        peak_memory_bytes: int = 128,
        wall_time_seconds: float = 1.0,
        cpu_time_seconds: float = 0.5,
    ) -> None:
        self.binding = binding
        self.improve = improve
        self.peak_memory_bytes = peak_memory_bytes
        self.wall_time_seconds = wall_time_seconds
        self.cpu_time_seconds = cpu_time_seconds
        manager = _shared_manager()
        self.call_count = manager.Value("i", 0)
        self._call_lock = manager.Lock()

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        self._record_call()
        return self._result(request)

    def _record_call(self) -> int:
        with self._call_lock:
            self.call_count.value += 1
            return cast(int, self.call_count.value)

    def _result(self, request: EvaluationRequest) -> RepeatEvidence:
        iterations = cast(int, request.parameters["iterations"])
        accuracy = 0.80 if iterations == 2 else (0.92 if self.improve else 0.80)
        return RepeatEvidence(
            identity=request.identity,
            status="succeeded",
            run_id=request.run_id,
            seed=request.seed,
            dispatch_digest=request.digest,
            granted_wall_time_seconds=request.remaining_wall_time_seconds,
            granted_cpu_time_seconds=request.remaining_cpu_time_seconds,
            granted_peak_memory_bytes=request.max_peak_memory_bytes,
            metrics={ACCURACY: accuracy, MEMORY: 1.0},
            wall_time_seconds=self.wall_time_seconds,
            cpu_time_seconds=self.cpu_time_seconds,
            peak_memory_bytes=self.peak_memory_bytes,
        )


class CrashingEvaluatorRunner(FakeEvaluatorRunner):
    def __init__(self, binding: EvaluatorBinding, *, crash_on_call: int) -> None:
        super().__init__(binding)
        self.crash_on_call = crash_on_call

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        if self._record_call() == self.crash_on_call:
            raise KeyboardInterrupt("synthetic process crash")
        return self._result(request)


class UncooperativeEvaluatorRunner(FakeEvaluatorRunner):
    def __init__(self, binding: EvaluatorBinding) -> None:
        super().__init__(binding)
        manager = _shared_manager()
        self.delayed_side_effect = manager.Value("i", 0)
        self._side_effect_lock = manager.Lock()

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        time.sleep(5)
        with self._side_effect_lock:
            self.delayed_side_effect.value += 1
        return super().evaluate(request)


class SecretFailingEvaluatorRunner(FakeEvaluatorRunner):
    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        self._record_call()
        raise RuntimeError("private-token-must-not-be-persisted")


class MismatchedEvaluatorRunner(FakeEvaluatorRunner):
    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        completed = super().evaluate(request)
        return completed.model_copy(update={"run_id": "run-wrong-completion"})


class CpuBurnerEvaluatorRunner(FakeEvaluatorRunner):
    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        self._record_call()
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes\n"
                "ctypes.CDLL(None).prctl(15, b'cpu ) burner', 0, 0, 0)\n"
                "value = 0\nwhile True:\n value = (value + 1) % 1000003",
            ],
            check=False,
        )
        raise RuntimeError("CPU-burning descendant unexpectedly exited")


class MemoryBurnerEvaluatorRunner(FakeEvaluatorRunner):
    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        self._record_call()
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes, time\n"
                "ctypes.CDLL(None).prctl(15, b'mem ) burner', 0, 0, 0)\n"
                "payload = bytearray(64 * 1024 * 1024)\ntime.sleep(60)",
            ],
            check=False,
        )
        raise RuntimeError("memory-burning descendant unexpectedly exited")


class BackgroundSideEffectEvaluatorRunner(FakeEvaluatorRunner):
    def __init__(self, binding: EvaluatorBinding, side_effect_path: Path) -> None:
        super().__init__(binding)
        self.side_effect_path = side_effect_path

    def evaluate(self, request: EvaluationRequest) -> RepeatEvidence:
        completed = super().evaluate(request)
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib, time\n"
                "time.sleep(0.5)\n"
                f"pathlib.Path({str(self.side_effect_path)!r}).write_text('escaped')",
            ]
        )
        return completed


__all__ = [
    "ACCURACY",
    "MEMORY",
    "MEMORY_LIMIT",
    "BackgroundSideEffectEvaluatorRunner",
    "CpuBurnerEvaluatorRunner",
    "CrashingEvaluatorRunner",
    "FakeEvaluatorRunner",
    "MemoryBurnerEvaluatorRunner",
    "MismatchedEvaluatorRunner",
    "SecretFailingEvaluatorRunner",
    "UncooperativeEvaluatorRunner",
]
