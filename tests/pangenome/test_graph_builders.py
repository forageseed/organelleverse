"""AC-001 / AC-002 — minigraph builds (and fails) through the controlled runner.

``build_graph(method="minigraph")`` must spawn the backend through the
process boundary in ``organelleverse.pangenome._runner``: an argument vector
with no shell redirection token, the staging ``pangenome.gfa`` handed over as
the stdout sink, and ``graph_built`` plus a single GFA ``ArtifactRef`` only
after the produced graph validates. Injection is private — the fake runner
replaces ``organelleverse.pangenome.pangenome.run_command`` so the public
signature stays legacy-compatible. Every failure mode (non-zero exit, spawn
error, empty or structurally invalid GFA) must close as ``status="failed"``
with a stable machine-readable error code and never report ``graph_built``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import organelleverse.pangenome.pangenome as pangenome_module
from organelleverse.core.artifacts import ArtifactRef
from organelleverse.core.genome import OrganelleGenome, OrganelleMetadata
from organelleverse.core.result import OrganelleResult
from organelleverse.pangenome.pangenome import build_graph
from organelleverse.pangenome._runner import CommandRecord

#: Smallest GFA that passes structural validation: one header, one segment.
MINIMAL_GFA = "H\tVN:Z:1.1\nS\tseg1\tACGTACGTACGTACGTACGTACGT\n"

#: Parses as GFA but links reference segments that were never declared.
INVALID_GFA = "H\tVN:Z:1.1\nL\tseg1\t+\tseg2\t+\t0M\n"


def _genome(tmp_path: Path, name: str, seq: str) -> OrganelleGenome:
    """Build a canonical genome whose sequence artifact exists on disk."""
    fasta = tmp_path / f"{name}.fa"
    fasta.write_text(f">{name}\n{seq}\n")
    return OrganelleGenome(
        organelle="mitochondrion",
        sequence=ArtifactRef.from_path(fasta, kind="sequence", format="fasta"),
        metadata=OrganelleMetadata(species=name, source="test"),
    )


def _two_genomes(tmp_path: Path) -> list[OrganelleGenome]:
    """A reference plus one alternate sample, both with FASTAs on disk."""
    return [
        _genome(tmp_path, "ref_sample", "ACGTACGTACGTACGTACGTACGTACGTACGTACGT"),
        _genome(tmp_path, "alt_sample", "ACGTACGTACGTACGTACGTACGTACGTACGTACGA"),
    ]


def test_minigraph_runs_through_runner_and_publishes_gfa_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        calls.append({"argv": list(argv), "stdout_path": stdout_path, "cwd": cwd})
        Path(stdout_path).write_text(MINIMAL_GFA)
        return CommandRecord(
            argv=tuple(argv), returncode=0, stdout_path=str(stdout_path), stderr=""
        )

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    genomes = _two_genomes(tmp_path)

    result = build_graph(
        genomes,
        output_dir=tmp_path / "out",
        method="minigraph",
    )

    assert len(calls) == 1, "minigraph must be spawned exactly once through the runner"
    argv = calls[0]["argv"]
    assert ">" not in argv, f"argv must not contain a shell redirection token: {argv}"
    assert argv[0].endswith("minigraph")
    assert "-xggs" in argv
    assert "-xgs" not in argv
    for fasta in (str(tmp_path / "ref_sample.fa"), str(tmp_path / "alt_sample.fa")):
        assert fasta in argv
    assert calls[0]["stdout_path"].endswith("pangenome.gfa")

    assert result.status == "ok"
    assert "graph_built" in result.flags
    assert "graph_planned" not in result.flags
    assert len([a for a in result.artifacts if a.kind == "pangenome_graph"]) == 1
    assert result.provenance.software_versions["minigraph"] == "fixture-version"
    artifact = result.artifacts[0]
    assert artifact.kind == "pangenome_graph"
    assert artifact.format == "gfa"
    assert artifact.uri.endswith("pangenome.gfa")
    assert Path(artifact.uri).is_file()


# -- AC-002: every failure mode closes as failed, never graph_built ---------


def _assert_failed_closed(result: OrganelleResult, *, code: str) -> None:
    """Assert a fail-closed result: failed status, stable code, no graph."""
    assert result.status == "failed", result.summary_text
    assert "graph_built" not in result.flags
    assert "graph_planned" not in result.flags
    assert result.artifacts == ()
    (error,) = result.errors
    assert error.code == code


def test_minigraph_nonzero_exit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero exit fails the build even when the emitted GFA looks valid."""

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        Path(stdout_path).write_text(MINIMAL_GFA)
        return CommandRecord(
            argv=tuple(argv), returncode=3, stdout_path=str(stdout_path), stderr="boom"
        )

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))

    result = build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="minigraph")

    _assert_failed_closed(result, code="pangenome.backend_failed")
    error = result.errors[0]
    assert error.details["returncode"] == 3
    assert error.details["stderr"] == "boom"


def test_minigraph_spawn_oserror_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn failure (e.g. missing binary) is normalized, never raised."""

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        raise FileNotFoundError(2, "No such file or directory", "minigraph")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))

    result = build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="minigraph")

    _assert_failed_closed(result, code="pangenome.backend_spawn_failed")
    assert result.errors[0].details["error"] == str(
        FileNotFoundError(2, "No such file or directory", "minigraph")
    )


def test_minigraph_empty_gfa_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 0 with a zero-byte GFA is not a graph."""

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        Path(stdout_path).write_text("")
        return CommandRecord(
            argv=tuple(argv), returncode=0, stdout_path=str(stdout_path), stderr=""
        )

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))

    result = build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="minigraph")

    _assert_failed_closed(result, code="pangenome.invalid_gfa")
    assert result.errors[0].details["path"].endswith("pangenome.gfa")


def test_minigraph_structurally_invalid_gfa_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 with a GFA that fails structural validation is not a graph."""

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        Path(stdout_path).write_text(INVALID_GFA)
        return CommandRecord(
            argv=tuple(argv), returncode=0, stdout_path=str(stdout_path), stderr=""
        )

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))

    result = build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="minigraph")

    _assert_failed_closed(result, code="pangenome.invalid_gfa")
    assert result.errors[0].details["path"].endswith("pangenome.gfa")


# -- AC-003: pggb spawns through the runner with numeric identity input -------


def test_pggb_uses_numeric_identity_pansn_input_and_discovers_final_gfa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pggb must receive a PAN-SN FASTA and numeric -p, then publish the .final.gfa.

    pggb is reference-free: the builder concatenates every haplotype into one
    FASTA whose headers carry the PAN-SN separator (exactly two ``#``), passes
    the percent identity as a numeric ``-p`` value, and afterwards discovers
    the ``*.smooth.final.gfa`` the pipeline deposits under its output tree
    instead of a fixed filename.
    """
    calls: list[dict[str, Any]] = []

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        argv_list = [str(item) for item in argv]
        calls.append({"argv": argv_list, "stdout_path": stdout_path, "cwd": cwd})
        if Path(argv_list[0]).name == "samtools":
            Path(f"{argv_list[-1]}.fai").write_text("index\n", encoding="utf-8")
            return CommandRecord(argv=tuple(argv_list), returncode=0, stdout_path=None, stderr="")
        # Emulate pggb writing its smoothed graph inside the output directory.
        final_gfa = Path(cwd) / "nested" / "test.smooth.final.gfa"
        final_gfa.parent.mkdir(parents=True, exist_ok=True)
        final_gfa.write_text(MINIMAL_GFA)
        return CommandRecord(argv=tuple(argv_list), returncode=0, stdout_path=None, stderr="")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    genomes = _two_genomes(tmp_path)

    result = build_graph(genomes, output_dir=tmp_path / "out", method="pggb")

    assert len(calls) == 2, "faidx and pggb must each run once through the runner"
    index_argv = calls[0]["argv"]
    assert Path(index_argv[0]).name == "samtools"
    assert index_argv[1] == "faidx"
    argv = calls[1]["argv"]
    identity = argv[argv.index("-p") + 1]
    assert identity == "90", f"-p must carry the numeric percent identity: {argv}"
    assert float(identity) == 90.0, f"-p value must be numeric, got {identity!r}"

    input_path = Path(argv[argv.index("-i") + 1])
    headers = [
        line[1:].strip() for line in input_path.read_text().splitlines() if line.startswith(">")
    ]
    assert headers == ["ref_sample#1#1", "alt_sample#1#1"]

    manifest_path = input_path.with_name("manifest.json")
    assert manifest_path.is_file(), "the same project contract must emit an audit manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["headers"] for item in manifest["samples"]] == [
        ["ref_sample#1#1"],
        ["alt_sample#1#1"],
    ]

    assert result.status == "ok"
    assert "graph_built" in result.flags
    final_artifacts = [a for a in result.artifacts if a.uri.endswith(".smooth.final.gfa")]
    assert len(final_artifacts) == 1, (
        f"exactly one .smooth.final.gfa artifact expected, got {result.artifacts}"
    )


# -- AC-004: pggb discovery ambiguity / unusable final graphs fail closed -----

#: Bytes that cannot decode as UTF-8 text under any locale.
BINARY_GFA = b"\xff\xfe\xfd\xfc\xfb not a gfa "


def _run_pggb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deposit: Callable[[Path], None],
) -> OrganelleResult:
    """Run a pggb build whose fake backend exits 0 after ``deposit(cwd)``.

    ``deposit`` emulates what pggb leaves under its output directory (``cwd``),
    so each test controls exactly what the final-GFA discovery will find.
    """

    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        argv_list = [str(item) for item in argv]
        if Path(argv_list[0]).name == "samtools":
            Path(f"{argv_list[-1]}.fai").write_text("index\n", encoding="utf-8")
            return CommandRecord(tuple(argv_list), 0, None, "")
        deposit(Path(cwd))
        return CommandRecord(argv=tuple(argv_list), returncode=0, stdout_path=None, stderr="")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    return build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="pggb")


def test_pggb_zero_final_gfa_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 0 with no *.smooth.final.gfa anywhere is a discovery failure."""

    result = _run_pggb(tmp_path, monkeypatch, deposit=lambda cwd: None)

    _assert_failed_closed(result, code="pangenome.final_gfa_missing")
    assert result.errors[0].details["search_root"].endswith("pggb")


def test_pggb_nonzero_exit_preserves_bounded_stdout_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_command(
        argv: list[str] | tuple[str, ...],
        *,
        stdout_path: str | Path | None = None,
        cwd: str | Path | None = None,
    ) -> CommandRecord:
        argv_list = [str(item) for item in argv]
        if Path(argv_list[0]).name == "samtools":
            Path(f"{argv_list[-1]}.fai").write_text("index\n", encoding="utf-8")
            return CommandRecord(tuple(argv_list), 0, None, "")
        assert stdout_path is not None
        Path(stdout_path).write_text("missing tool: wfmash\n", encoding="utf-8")
        return CommandRecord(tuple(argv), 1, str(stdout_path), "wfmash: reference missing\n")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    result = build_graph(_two_genomes(tmp_path), output_dir=tmp_path / "out", method="pggb")

    _assert_failed_closed(result, code="pangenome.backend_failed")
    assert result.errors[0].details["stdout_tail"] == "missing tool: wfmash\n"
    assert "wfmash: reference missing" in result.errors[0].message


def test_pggb_multiple_nested_final_gfa_fails_closed_with_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 0 with two nested finals is ambiguous: failed, candidates reported."""

    relative = ("run1/a.smooth.final.gfa", "deep/nested/b.smooth.final.gfa")

    def deposit(cwd: Path) -> None:
        for rel in relative:
            final = cwd / rel
            final.parent.mkdir(parents=True, exist_ok=True)
            final.write_text(MINIMAL_GFA)

    result = _run_pggb(tmp_path, monkeypatch, deposit)

    _assert_failed_closed(result, code="pangenome.final_gfa_ambiguous")
    expected = sorted(str(tmp_path / "out" / "pggb" / rel) for rel in relative)
    assert sorted(result.errors[0].details["matches"]) == expected, result.errors[0].details


@pytest.mark.parametrize(
    ("label", "content"),
    [
        pytest.param("empty-gfa", "", id="empty"),
        pytest.param("structurally-invalid-gfa", INVALID_GFA, id="structurally-invalid"),
        pytest.param("invalid-utf8-bytes", BINARY_GFA, id="invalid-utf8"),
    ],
)
def test_pggb_single_unusable_final_gfa_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, content: str | bytes
) -> None:
    """One discovered final that is empty, malformed, or undecodable is no graph."""

    def deposit(cwd: Path) -> None:
        final = cwd / "test.smooth.final.gfa"
        if isinstance(content, str):
            final.write_text(content)
        else:
            final.write_bytes(content)

    result = _run_pggb(tmp_path, monkeypatch, deposit)

    _assert_failed_closed(result, code="pangenome.invalid_gfa")
    assert result.errors[0].details["path"].endswith(".smooth.final.gfa")


def test_pggb_unreadable_final_gfa_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final whose read raises OSError fails closed instead of propagating."""

    def deposit(cwd: Path) -> None:
        final = cwd / "test.smooth.final.gfa"
        final.write_text(MINIMAL_GFA)
        final.chmod(0o000)
        if os.access(final, os.R_OK):  # e.g. running as root: force the read error

            def raise_oserror(path: str | Path) -> None:
                raise OSError("simulated unreadable GFA")

            monkeypatch.setattr(pangenome_module, "validate_gfa", raise_oserror)

    result = _run_pggb(tmp_path, monkeypatch, deposit)

    _assert_failed_closed(result, code="pangenome.invalid_gfa")
    assert result.errors[0].details["path"].endswith(".smooth.final.gfa")


# -- AC-005: unsupported/invalid requests fail before any process spawn ------


@pytest.mark.parametrize("method", ["unknown"])
def test_unverified_backend_is_rejected_without_invoking_legacy_executor(
    tmp_path: Path, method: str
) -> None:
    calls: list[list[str]] = []

    def executor(argv: list[str]) -> None:
        calls.append(argv)

    result = build_graph(
        _two_genomes(tmp_path),
        output_dir=tmp_path / "out",
        method=method,
        executor=executor,
    )

    _assert_failed_closed(result, code="pangenome.unsupported_backend")
    assert calls == []
    assert result.errors[0].details["supported"] == ("minigraph", "pggb", "pantools")


@pytest.mark.parametrize("reference_index", [-1, 2])
def test_minigraph_rejects_reference_index_outside_genome_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_index: int
) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_: object) -> CommandRecord:
        calls.append(argv)
        return CommandRecord(tuple(argv), 0, None, "")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    result = build_graph(
        _two_genomes(tmp_path),
        output_dir=tmp_path / "out",
        method="minigraph",
        reference_index=reference_index,
    )

    _assert_failed_closed(result, code="pangenome.invalid_parameter")
    assert calls == []
    assert result.errors[0].details["parameter"] == "reference_index"


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"k": 0}, "k"),
        ({"threads": 0}, "threads"),
        ({"n_haplotypes": 0}, "n_haplotypes"),
        ({"segment_length": 0}, "segment_length"),
        ({"identity": 0.0}, "identity"),
        ({"identity": 101.0}, "identity"),
    ],
)
def test_graph_builder_rejects_invalid_numeric_parameters_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, int | float],
    parameter: str,
) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_: object) -> CommandRecord:
        calls.append(argv)
        return CommandRecord(tuple(argv), 0, None, "")

    monkeypatch.setattr(pangenome_module, "run_command", _versioned_runner(fake_run_command))
    result = build_graph(
        _two_genomes(tmp_path),
        output_dir=tmp_path / "out",
        method="pggb",
        **kwargs,
    )

    _assert_failed_closed(result, code="pangenome.invalid_parameter")
    assert calls == []
    assert result.errors[0].details["parameter"] == parameter


def test_pggb_companion_resolution_falls_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = tmp_path / "isolated" / "pggb"
    backend.parent.mkdir()
    backend.write_text("launcher", encoding="utf-8")
    monkeypatch.setattr(pangenome_module.shutil, "which", lambda name: f"/usr/bin/{name}")

    resolved = pangenome_module._companion_executable("samtools", str(backend))

    assert resolved == "/usr/bin/samtools"


def test_installed_pantools_inventory_reports_versioned_export_contract(monkeypatch):
    from organelleverse.pangenome import install

    monkeypatch.setattr(install.shutil, "which", lambda cli: "/tools/pantools")
    result = install.check_backend("pantools", scan_envs=False)
    assert result["installed"] is True
    assert result["path"] == "/tools/pantools"
    assert "4.3.5 property-export adapter" in result["note"]
    assert "sample/molecule paths" in install.install_hint("pantools")


def _versioned_runner(build_runner):
    """Version response for mocked builders; real gates never use this fixture."""

    def run(argv, **kwargs):
        if argv[-1] == "--version":
            destination = Path(kwargs["stdout_path"])
            destination.write_text("fixture-version\n")
            return CommandRecord(tuple(argv), 0, str(destination), "")
        return build_runner(argv, **kwargs)

    return run


def test_version_probe_failure_is_not_published_as_a_reproducible_graph(tmp_path, monkeypatch):
    def run(argv, *, stdout_path=None, cwd=None):
        if argv[-1] == "--version":
            Path(stdout_path).write_text("")
            return CommandRecord(tuple(argv), 1, str(stdout_path), "version unavailable")
        Path(stdout_path).write_text(MINIMAL_GFA)
        return CommandRecord(tuple(argv), 0, str(stdout_path), "")

    monkeypatch.setattr(pangenome_module, "run_command", run)
    result = build_graph(_two_genomes(tmp_path), method="minigraph", output_dir=tmp_path / "out")
    assert result.status == "failed"
    assert result.errors[0].code == "pangenome.backend_version_failed"
    assert "graph_built" not in result.flags
    assert not result.artifacts
