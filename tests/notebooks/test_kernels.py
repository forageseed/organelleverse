"""Kernel session protocol tests over real subprocesses."""

from __future__ import annotations

import pytest

from organelleverse.core.errors import OrganelleDependencyError
from organelleverse.notebooks.kernels import KernelSession


def test_python_kernel_roundtrip_small_value_and_stdout() -> None:
    session = KernelSession("python")
    try:
        response = session.execute("print('hello')\n40 + 2")

        assert response.ok is True
        assert response.stdout == "hello\n"
        assert response.result == "42"
        assert response.error is None
    finally:
        session.shutdown()


def test_python_kernel_syntax_error_is_structured_not_fatal() -> None:
    session = KernelSession("python")
    try:
        broken = session.execute("def (")
        assert broken.ok is False
        assert broken.error is not None
        assert "SyntaxError" in broken.error

        healthy = session.execute("2 * 3")
        assert healthy.ok is True
        assert healthy.result == "6"
    finally:
        session.shutdown()


def test_declared_paths_reach_the_cell_namespace(tmp_path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    target = tmp_path / "output.json"
    session = KernelSession("python")
    try:
        response = session.execute(
            "import json\n"
            "rows = open(INPUT_PATHS['table']).read()\n"
            "open(OUTPUT_PATHS['summary'], 'w').write(json.dumps({'lines': rows.count(chr(10))}))\n"
            "'done'",
            inputs={"table": str(source)},
            outputs={"summary": str(target)},
        )
        assert response.ok is True, response.error
        assert target.read_text(encoding="utf-8") == '{"lines": 2}'
    finally:
        session.shutdown()


def test_shutdown_is_idempotent_and_restart_works() -> None:
    session = KernelSession("python")
    assert session.execute("1").ok is True
    session.shutdown()
    session.shutdown()
    assert session.alive() is False
    assert session.execute("2").ok is True
    session.shutdown()


def test_isolation_between_two_python_sessions() -> None:
    first = KernelSession("python")
    second = KernelSession("python")
    try:
        assert first.execute("shared_value = 99\nshared_value").result == "99"
        missing = second.execute("'shared_value' in dir()")
        assert missing.ok is True
        assert missing.result == "False"
    finally:
        first.shutdown()
        second.shutdown()


def test_broken_pipe_raises_structured_error() -> None:
    session = KernelSession("python")
    process = session._ensure_started()
    process.kill()
    process.wait()
    with pytest.raises(OrganelleDependencyError) as info:
        session.execute("1")
    assert info.value.code == "kernel.process_lost"


@pytest.mark.r_kernel
def test_r_kernel_roundtrip_and_structured_error() -> None:
    session = KernelSession("r")
    try:
        response = session.execute("cat('bonjour\\n')\n1 + 1")
        assert response.ok is True
        assert "bonjour" in response.stdout
        assert response.error is None

        broken = session.execute("stop('boom')")
        assert broken.ok is False
        assert "boom" in (broken.error or "")

        healthy = session.execute("cat('still-alive\\n')")
        assert healthy.ok is True
    finally:
        session.shutdown()


@pytest.mark.r_kernel
def test_r_kernel_isolation_from_python_session() -> None:
    python_session = KernelSession("python")
    r_session = KernelSession("r")
    try:
        python_session.execute("python_only = 5\npython_only")
        denied = r_session.execute("exists('python_only')")
        assert denied.ok is True
        assert "FALSE" in denied.stdout
    finally:
        python_session.shutdown()
        r_session.shutdown()


@pytest.mark.r_kernel
def test_r_kernel_binary_frames_preserve_unicode_and_path_manifests(tmp_path) -> None:
    session = KernelSession("r")
    try:
        response = session.execute(
            "cat('细胞器\\n')\nstopifnot(input.paths[['reads']] == output.paths[['copy']])",
            inputs={"reads": str(tmp_path / "reads=1.tsv")},
            outputs={"copy": str(tmp_path / "reads=1.tsv")},
        )
        assert response.ok is True, response.error
        assert response.stdout == "细胞器"
        assert session.execute("6 * 7").stdout == "[1] 42"
    finally:
        session.shutdown()
