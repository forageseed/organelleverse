"""Original-WGS nuclear coverage using unmodified GenomeScope 2.

This optional provider counts canonical raw k21, then uses the official model's
own fitting and model selection. It never selects a threshold using assembly
references, recruited organelle reads, or a replacement fit after failure.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from organelleverse.core.errors import OrganelleDependencyError

# PacBio defines HiFi as CCS with read QV >= 20. This is an input/model
# consistency requirement, not an empirical model-fit cutoff or accuracy proof.
HIFI_MIN_READ_QV = 20
HIFI_MAX_ERROR_PERCENT = 100 * 10 ** (-HIFI_MIN_READ_QV / 10)
NUCLEAR_COVERAGE_PROTOCOL_VERSION = "2"


@dataclass(frozen=True)
class CoverageTools:
    jellyfish: Path
    rscript: Path
    genomescope: Path

    def provenance(self) -> dict[str, str]:
        commands = {
            "jellyfish": [str(self.jellyfish), "--version"],
            "Rscript": [str(self.rscript), "--version"],
            "genomescope": [str(self.rscript), str(self.genomescope), "-v"],
            "R_packages": [str(self.rscript), "-e",
                           "for (p in c('genomescope','minpack.lm','argparse','jsonlite')) "
                           "cat(p, as.character(packageVersion(p)), "
                           "normalizePath(find.package(p)), '\\n')"],
        }
        report = {
            "nuclear_coverage_protocol_version": NUCLEAR_COVERAGE_PROTOCOL_VERSION,
            "jellyfish_path": str(self.jellyfish),
            "rscript_path": str(self.rscript),
            "genomescope_script": str(self.genomescope),
            "R_LIBS_USER": os.environ.get("R_LIBS_USER", ""),
        }
        for name, command in commands.items():
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        check=True, timeout=60, env=self.environment())
            except (OSError, subprocess.SubprocessError) as error:
                raise OrganelleDependencyError(
                    code="assembly.environment_unavailable",
                    message=f"Nuclear coverage provider {name} could not report its version: {error}",
                    details={"command": command},
                ) from error
            report[name + "_version"] = (result.stdout + result.stderr).strip()
        return report

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = str(self.rscript.parent) + os.pathsep + env.get("PATH", "")
        return env


def resolve_tools() -> CoverageTools:
    """Resolve installed providers; no download or silent substitute."""
    if sys.platform != "linux":
        raise ValueError("Automatic nuclear coverage requires the Linux scientific provider")
    paths = {}
    for name, variable in (
        ("jellyfish", "ORG_VERSE_JELLYFISH_BIN"),
        ("Rscript", "ORG_VERSE_RSCRIPT_BIN"),
        ("genomescope.R", "ORG_VERSE_GENOMESCOPE_SCRIPT"),
    ):
        value = os.environ.get(variable) or shutil.which(name)
        if not value or not Path(value).is_file():
            raise OrganelleDependencyError(
                code="assembly.environment_unavailable",
                message=f"Automatic nuclear coverage needs {name}; install it on the server "
                        f"and set {variable} or add it to PATH",
                details={"missing": [name]},
            )
        paths[name] = Path(value).resolve()
    return CoverageTools(paths["jellyfish"], paths["Rscript"], paths["genomescope.R"])


def parse_model(model: str, summary: str) -> dict:
    coefficient = re.search(r"^\s*kmercov\s+(\S+)\s+(\S+)", model, re.M)
    length = re.search(r"^Genome Haploid Length\s+(\S+)\s+bp\s+(\S+)\s+bp", summary, re.M)
    if coefficient is None or length is None:
        raise ValueError("No reported nuclear coverage coefficient or haploid-size interval")
    mean, error = map(float, coefficient.groups())
    size_min, size_max = (float(value.replace(",", "")) for value in length.groups())
    if not all(math.isfinite(value) for value in (mean, error, size_min, size_max)):
        raise ValueError("Unbounded or nonfinite model interval; no threshold selected")
    low, high = mean - 1.96 * error, mean + 1.96 * error
    if error < 0 or low <= 0 or size_min <= 0 or size_max < size_min:
        raise ValueError("Nuclear coverage is not identified by a positive finite interval")
    fit = re.search(r"^\s*Model Fit\s+(\S+)%\s+(\S+)%", summary, re.M)
    read_error = re.search(r"^\s*Read Error Rate\s+(\S+)%\s+(\S+)%", summary, re.M)
    if fit is None or read_error is None:
        raise ValueError("No reported model-fit or read-error diagnostics")
    fit_all, fit_full = map(float, fit.groups())
    error_values = tuple(map(float, read_error.groups()))
    if not all(math.isfinite(value) for value in (fit_all, fit_full, *error_values)):
        raise ValueError("Nonfinite model-fit or read-error diagnostics")
    if not all(0 <= value <= HIFI_MAX_ERROR_PERCENT for value in error_values):
        raise ValueError(
            f"GenomeScope inferred read error {error_values}% contradicts declared HiFi "
            f"(QV >= {HIFI_MIN_READ_QV}, error <= {HIFI_MAX_ERROR_PERCENT}%); "
            "no nuclear threshold selected"
        )
    return {
        "nuclear_haplotype_kmer_coverage": mean,
        "standard_error": error,
        "approximate_95_percent_coverage_interval": [low, high],
        "haploid_size_interval_bp": [size_min, size_max],
        "graph_min_count": math.ceil(5 * mean),
        "threshold_interval_from_coverage": [math.ceil(5 * low), math.ceil(5 * high)],
        "coverage_or_assembly_accuracy_certified": False,
        "coefficient_precision": "as printed by the official model report",
        # GenomeScope's misleading min/max headers report two fitting domains,
        # not a confidence interval; its two read-error values repeat one point.
        "model_fit_all_percent": fit_all,
        "model_fit_full_percent": fit_full,
        "fitted_read_error_rate_percent": error_values[0],
        "declared_hifi_min_read_qv": HIFI_MIN_READ_QV,
        "model_input_consistency_checked": True,
    }


def fit_original_wgs(
    reads: Path, output: Path, *, threads: int, ploidy: int,
    tools: CoverageTools | None = None, log: Callable[[str], None] = lambda _: None,
) -> dict:
    """Count the supplied original WGS once and retain the model and histogram."""
    if not 1 <= ploidy <= 6 or not 1 <= threads <= 256:
        raise ValueError("GenomeScope needs ploidy in 1..6 and threads in 1..256")
    tools = tools or resolve_tools()
    output.mkdir(parents=True, exist_ok=True)
    database = output / "canonical-k21.jf"
    histogram = output / "histogram.tsv"
    fit = output / "fit"
    protocol = {
        "input_context": "original whole-genome reads before recruitment",
        "input": str(reads), "k": 21, "canonical": True, "ploidy": ploidy,
        "fitting_domain": "unmodified official GenomeScope defaults",
        "model": "GenomeScope 2 negative-binomial mixture",
        "cutoff_rule": "ceil(5 * fitted nuclear haplotype coverage)",
        "assembly_reference_used": False,
        "declared_input_quality": "PacBio HiFi, read QV >= 20; fitted error <= 1% required",
        "tools": tools.provenance(),
        "sources": ["https://doi.org/10.1038/s41467-020-14998-3",
                    "https://doi.org/10.1186/s13059-025-03676-6",
                    "https://pacbiofileformats.readthedocs.io/en/12.0/BAM.html#hifi-reads"],
    }
    commands = [
        [str(tools.jellyfish), "count", "-C", "-m", "21", "-s", "100M",
         "-t", str(threads), "-o", str(database), "/dev/fd/0"],
        [str(tools.jellyfish), "histo", "-t", str(threads), "-h", "1000000", str(database)],
        [str(tools.rscript), str(tools.genomescope), "-i", str(histogram),
         "-o", str(fit), "-k", "21", "-p", str(ploidy)],
    ]
    protocol["commands"] = commands
    profile_path = output / "coverage-profile.json"
    profile = {"status": "running", "protocol": protocol}
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    try:
        log("OVASM: counting canonical raw k21 from original WGS for nuclear coverage")
        with (output / "count.log").open("wb") as count_log:
            process = subprocess.Popen(commands[0], stdin=subprocess.PIPE, stdout=count_log,
                                       stderr=count_log, env=tools.environment())
            pipe_error = None
            try:
                with reads.open("rb") as check:
                    compressed = check.read(2) == b"\x1f\x8b"
                opener = gzip.open if compressed else Path.open
                with opener(reads, "rb") as source:
                    shutil.copyfileobj(source, process.stdin, 1024 * 1024)
            except BrokenPipeError as error:
                pipe_error = error
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError as error:
                    pipe_error = pipe_error or error
                finally:
                    returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, commands[0])
            if pipe_error is not None:
                raise pipe_error
            with histogram.open("wb") as target:
                subprocess.run(commands[1], stdout=target, stderr=count_log,
                               check=True, env=tools.environment())
        log("OVASM: fitting the official GenomeScope model; no threshold if the interval is unidentified")
        with (output / "fit.log").open("wb") as fit_log:
            subprocess.run(commands[2], stdout=fit_log, stderr=fit_log,
                           check=True, env=tools.environment())
        profile.update(parse_model((fit / "model.txt").read_text(),
                                   (fit / "summary.txt").read_text()), status="completed")
    except Exception as error:
        profile.update(status="failed", error=f"{type(error).__name__}: {error}")
        profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        raise
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    return profile
