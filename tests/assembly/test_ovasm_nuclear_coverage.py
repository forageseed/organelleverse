"""Automatic nuclear fitting must use original WGS and expose identification failure."""
from types import SimpleNamespace

import pytest

from organelleverse.assembly import nuclear_coverage
from organelleverse.assembly.contracts import OvasmParameters
from organelleverse.core.errors import OrganelleInputError
from organelleverse.assembly import ovasm_pipeline as ovasm_runner
from tests.assembly.test_ovasm_backend import _context, _long


def test_automatic_threshold_requires_original_hifi_wgs(tmp_path):
    data = _long(tmp_path)
    context = _context(tmp_path, data, backend_parameters=OvasmParameters(
        graph_mode="multichrom", graph_min_count="auto", nuclear_ploidy=4,
    ))
    assert context.effective_backend_parameters["nuclear_ploidy"] == 4
    with pytest.raises(OrganelleInputError, match="whole_genome"):
        _context(tmp_path, data, backend_parameters=OvasmParameters(
            read_set="target_reads", graph_mode="multichrom", graph_min_count="auto",
        ))
    with pytest.raises(OrganelleInputError, match="HiFi"):
        _context(tmp_path, _long(tmp_path, "ont", "raw"), backend_parameters=OvasmParameters(
            graph_mode="multichrom", graph_min_count="auto",
        ))


def test_failed_model_stops_before_recruitment_or_reference_loading(tmp_path, monkeypatch):
    original = tmp_path / "raw.fastq"
    original.write_text("@r\nACGT\n+\nIIII\n")
    calls = []

    def fail(reads, output, **kwargs):
        calls.append((reads, output, kwargs["ploidy"]))
        raise ValueError("Unbounded model interval")

    monkeypatch.setattr(nuclear_coverage, "fit_original_wgs", fail)
    monkeypatch.setattr(ovasm_runner._ovasm, "run_recruit",
                        lambda *args, **kwargs: pytest.fail("Recruited before nuclear model"))
    with pytest.raises(ValueError, match="Unbounded"):
        ovasm_runner.run_ovasm(
            {"strategy": "hifi_only", "graph_mode": "multichrom",
             "graph_min_count": "auto", "nuclear_ploidy": 4},
            organelle="mitochondrion", reads=original, output_dir=tmp_path / "result",
            log=lambda _: None,
        )
    assert calls == [(original, tmp_path / "result/nuclear-coverage", 4)]


@pytest.mark.parametrize("sizes", ["1,000 bp Inf bp", "0 bp 1,000 bp", "2,000 bp 1,000 bp"])
def test_convergence_without_identifiable_genome_size_is_rejected(sizes):
    with pytest.raises(ValueError):
        nuclear_coverage.parse_model("kmercov 15.96 0.01737", "Genome Haploid Length " + sizes)


def test_threshold_records_the_interval_separately_from_accuracy():
    profile = nuclear_coverage.parse_model(
        "kmercov 3.601 0.05902",
        "Genome Haploid Length 340,036,397 bp 363,082,335 bp\n"
        "Model Fit 90% 99%\nRead Error Rate 0.1% 0.1%",
    )
    assert profile["graph_min_count"] == 19
    assert profile["threshold_interval_from_coverage"] == [18, 19]
    assert profile["coverage_or_assembly_accuracy_certified"] is False
    assert profile["model_fit_all_percent"] == 90
    assert profile["model_fit_full_percent"] == 99
    assert profile["fitted_read_error_rate_percent"] == 0.1


@pytest.mark.requires_posix
def test_missing_model_provider_is_a_dependency_error(monkeypatch):
    from organelleverse.core.errors import OrganelleDependencyError

    for variable in ("ORG_VERSE_JELLYFISH_BIN", "ORG_VERSE_RSCRIPT_BIN", "ORG_VERSE_GENOMESCOPE_SCRIPT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(nuclear_coverage.shutil, "which", lambda _: None)
    with pytest.raises(OrganelleDependencyError, match="jellyfish"):
        nuclear_coverage.resolve_tools()


def test_provider_version_changes_existing_environment_identity(tmp_path, monkeypatch):
    from organelleverse.assembly.backends import ovasm as backend
    from organelleverse.assembly.environment_specs import OVASM_ENVIRONMENT

    binary = tmp_path / "ovasm"
    binary.write_bytes(b"existing native binary")
    monkeypatch.setattr(backend, "resolve_ovasm", lambda: binary)
    version = {"genomescope_version": "GenomeScope 2.0.1"}
    tools = SimpleNamespace(provenance=lambda: dict(version))
    monkeypatch.setattr(nuclear_coverage, "resolve_tools", lambda: tools)
    ordinary_before = backend.NativeOvasmEnvironment().expected_environment_digest(OVASM_ENVIRONMENT)
    first = backend.NativeOvasmEnvironment({"graph_min_count": "auto"})
    first_digest = first.expected_environment_digest(OVASM_ENVIRONMENT)
    version["genomescope_version"] = "a different reported model version"
    second = backend.NativeOvasmEnvironment({"graph_min_count": "auto"})
    assert first_digest != second.expected_environment_digest(OVASM_ENVIRONMENT)
    assert ordinary_before == backend.NativeOvasmEnvironment().expected_environment_digest(OVASM_ENVIRONMENT)
    version["nuclear_coverage_protocol_version"] = "2"
    third = backend.NativeOvasmEnvironment({"graph_min_count": "auto"})
    assert second.expected_environment_digest(OVASM_ENVIRONMENT) != third.expected_environment_digest(OVASM_ENVIRONMENT)


def test_automatic_command_uses_prepared_provider_paths(tmp_path):
    from organelleverse.assembly.backends.ovasm import OvasmAdapter
    from organelleverse.assembly.environments import PreparedExecutable

    context = _context(tmp_path, _long(tmp_path), backend_parameters=OvasmParameters(
        graph_mode="multichrom", graph_min_count="auto", nuclear_ploidy=4,
    ))
    dependencies = tuple(PreparedExecutable(name=name, path=tmp_path / name)
                         for name in ("jellyfish", "Rscript", "genomescope.R"))
    context = context.model_copy(update={"environment": context.environment.model_copy(update={
        "executables": (*context.environment.executables, *dependencies),
    })})
    command = OvasmAdapter().build_command(context)
    assert command.stable_argv[command.stable_argv.index("--graph-min-count") + 1] == "auto"
    assert command.stable_argv[command.stable_argv.index("--nuclear-ploidy") + 1] == "4"
    assert command.resolved_argv[command.resolved_argv.index("--coverage-genomescope") + 1] == str(tmp_path / "genomescope.R")


def test_early_counter_exit_is_reaped_and_reported_before_fitting(tmp_path, monkeypatch):
    import json
    import subprocess

    original = tmp_path / "raw.fastq"
    original.write_text("@r\nACGT\n+\nIIII\n")
    waited = []

    class ClosedCounterPipe:
        def write(self, data):
            raise BrokenPipeError("counter exited before consuming input")

        def close(self):
            raise BrokenPipeError("buffer flush after counter exit")

    counter = SimpleNamespace(stdin=ClosedCounterPipe(), wait=lambda: waited.append(True) or -15)
    tools = SimpleNamespace(jellyfish=tmp_path / "counter", rscript=tmp_path / "Rscript",
                            genomescope=tmp_path / "genomescope.R",
                            provenance=lambda: {}, environment=lambda: {})
    monkeypatch.setattr(nuclear_coverage.subprocess, "Popen", lambda *args, **kwargs: counter)
    monkeypatch.setattr(nuclear_coverage.subprocess, "run",
                        lambda *args, **kwargs: pytest.fail("Fitting after counter failure"))
    with pytest.raises(subprocess.CalledProcessError) as error:
        nuclear_coverage.fit_original_wgs(original, tmp_path / "profile", threads=1,
                                         ploidy=2, tools=tools)
    assert error.value.returncode == -15
    assert waited == [True]
    profile = json.loads((tmp_path / "profile/coverage-profile.json").read_text())
    assert profile["status"] == "failed"
    assert "CalledProcessError" in profile["error"]


def test_bounded_fit_conflicting_with_declared_hifi_is_rejected():
    with pytest.raises(ValueError, match="contradicts declared HiFi"):
        nuclear_coverage.parse_model(
            "kmercov 38.7 0.6",
            "Genome Haploid Length 4,381,174 bp 4,587,295 bp\n"
            "Model Fit 43.5455% 84.0967%\nRead Error Rate 7.38565% 7.38565%",
        )


def test_fitting_protocol_version_is_reported_with_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(nuclear_coverage.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout="provider version", stderr=""))
    tools = nuclear_coverage.CoverageTools(tmp_path / "jellyfish", tmp_path / "Rscript",
                                         tmp_path / "genomescope.R")
    assert tools.provenance()["nuclear_coverage_protocol_version"] == "2"
