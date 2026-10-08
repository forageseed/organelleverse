"""Deterministic, test-only real-data fixtures for the HiMT release gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from Bio import SeqIO
from pydantic import BaseModel, ConfigDict, Field, model_validator

from organelleverse.assembly.environment_specs import HIMT_ENVIRONMENT
from organelleverse.assembly.environments import EnvironmentManager
from organelleverse.core.errors import OrganelleDependencyError

FixtureProfile = Literal["hifi", "clr", "ont"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class FixtureSource(_StrictModel):
    order: Annotated[int, Field(ge=0)]
    accession: Annotated[str, Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    size_bytes: Annotated[int, Field(ge=1)]
    md5: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    records: Annotated[int, Field(ge=1)] | None
    platform: Annotated[str, Field(min_length=1)]
    library_strategy: Annotated[str, Field(min_length=1)]


class FixtureReference(_StrictModel):
    accessions: Annotated[tuple[str, ...], Field(min_length=1)]
    url: Annotated[str, Field(min_length=1)]
    size_bytes: Annotated[int, Field(ge=1)]
    md5: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SelectionRecipe(_StrictModel):
    mapper: Literal["minimap2"]
    preset: Literal["map-pb", "map-ont"]
    threads: Literal[1]
    min_mapq: Annotated[int, Field(ge=0, le=255)]
    min_aligned_span: Annotated[int, Field(ge=1)]
    id_format: Literal["{accession}|{original_id}"]


class ProfileFixture(_StrictModel):
    profile: FixtureProfile
    technology: Literal["pacbio_hifi", "pacbio_clr", "ont"]
    quality_state: Literal["ccs", "raw"]
    biosample: Annotated[str, Field(min_length=1)]
    study: Annotated[str, Field(min_length=1)]
    platform: Annotated[str, Field(min_length=1)]
    license: Annotated[str, Field(min_length=1)]
    sources: Annotated[tuple[FixtureSource, ...], Field(min_length=1)]
    reference: FixtureReference | None
    selection: SelectionRecipe | None

    @model_validator(mode="after")
    def _validate_profile(self) -> Self:
        expected = {
            "hifi": ("pacbio_hifi", "ccs"),
            "clr": ("pacbio_clr", "raw"),
            "ont": ("ont", "raw"),
        }[self.profile]
        if (self.technology, self.quality_state) != expected:
            raise ValueError("fixture profile technology/quality mismatch")
        if tuple(source.order for source in self.sources) != tuple(range(len(self.sources))):
            raise ValueError("fixture source order must be contiguous and canonical")
        accessions = tuple(source.accession for source in self.sources)
        if len(set(accessions)) != len(accessions):
            raise ValueError("fixture source accessions must be unique")
        if self.profile == "hifi":
            if len(self.sources) != 1 or self.reference is not None or self.selection is not None:
                raise ValueError("HiFi fixture must have one source and no selection recipe")
        elif self.reference is None or self.selection is None:
            raise ValueError("CLR/ONT fixture requires a reference and selection recipe")
        elif self.selection.preset != ("map-ont" if self.profile == "ont" else "map-pb"):
            raise ValueError("fixture minimap2 preset does not match profile")
        return self


class FixtureManifest(_StrictModel):
    profiles: Annotated[tuple[ProfileFixture, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _validate_profiles(self) -> Self:
        names = tuple(profile.profile for profile in self.profiles)
        if len(set(names)) != len(names):
            raise ValueError("fixture profiles must be unique")
        return self


class SourceIdentity(_StrictModel):
    order: int
    accession: str
    url: str
    size_bytes: int
    md5: str
    sha256: str


class ReferenceIdentity(_StrictModel):
    accessions: tuple[str, ...]
    url: str
    size_bytes: int
    md5: str
    sha256: str


class PreparedFixture(_StrictModel):
    profile: FixtureProfile
    technology: Literal["pacbio_hifi", "pacbio_clr", "ont"]
    quality_state: Literal["ccs", "raw"]
    path: Path
    sha256: str
    size_bytes: int
    records: int
    source_identities: tuple[SourceIdentity, ...]
    reference_identity: ReferenceIdentity | None
    recipe_identity: str
    minimap2_version: str
    minimap2_argv: tuple[tuple[str, ...], ...]


class _DerivationManifest(_StrictModel):
    profile: FixtureProfile
    sha256: str
    size_bytes: int
    records: int
    source_identities: tuple[SourceIdentity, ...]
    reference_identity: ReferenceIdentity | None
    recipe_identity: str
    minimap2_version: str
    minimap2_argv: tuple[tuple[str, ...], ...]
    output_filename: str


class _LocalFile(_StrictModel):
    path: Path
    size_bytes: int
    md5: str
    sha256: str


def _source_manifest_path() -> Path:
    return Path(__file__).parent / "fixtures" / "assembly" / "himt" / "sources.json"


def _expected_directory() -> Path:
    return Path(__file__).parent / "fixtures" / "assembly" / "himt"


def load_fixture_manifest(path: Path | None = None) -> FixtureManifest:
    source = path or _source_manifest_path()
    return FixtureManifest.model_validate_json(source.read_bytes())


def _profile_from_manifest(manifest: FixtureManifest, profile: FixtureProfile) -> ProfileFixture:
    for candidate in manifest.profiles:
        if candidate.profile == profile:
            return candidate
    raise ValueError(f"fixture manifest does not define profile {profile!r}")


def _hash_file(path: Path) -> _LocalFile:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return _LocalFile(
        path=path,
        size_bytes=size,
        md5=md5.hexdigest(),
        sha256=sha256.hexdigest(),
    )


def _verify_file(
    path: Path,
    *,
    expected_size: int,
    expected_md5: str | None,
    expected_sha256: str | None,
    label: str,
) -> _LocalFile:
    actual = _hash_file(path)
    if actual.size_bytes != expected_size:
        raise OrganelleDependencyError(
            code="fixture.source_size_mismatch",
            message=f"{label} byte size does not match its source manifest",
            details={"expected": expected_size, "actual": actual.size_bytes},
        )
    if expected_md5 is not None and actual.md5 != expected_md5:
        raise OrganelleDependencyError(
            code="fixture.source_md5_mismatch",
            message=f"{label} MD5 does not match its source manifest",
            details={"expected": expected_md5, "actual": actual.md5},
        )
    if expected_sha256 is not None and actual.sha256 != expected_sha256:
        raise OrganelleDependencyError(
            code="fixture.source_sha256_mismatch",
            message=f"{label} SHA256 does not match its source manifest",
            details={"expected": expected_sha256, "actual": actual.sha256},
        )
    return actual


def _cache_filename(accession: str, url: str) -> str:
    suffixes = "".join(Path(urlparse(url).path).suffixes)
    if suffixes not in {
        ".fa",
        ".fasta",
        ".fq",
        ".fastq",
        ".fa.gz",
        ".fasta.gz",
        ".fq.gz",
        ".fastq.gz",
    }:
        suffixes = ".fasta"
    return f"{accession}{suffixes}"


def _download_verified(
    *,
    cache_root: Path,
    accession: str,
    url: str,
    size_bytes: int,
    md5: str,
    sha256: str | None,
) -> _LocalFile:
    declared_identity = sha256 or md5
    destination = (
        cache_root
        / "fixtures"
        / "himt"
        / "sources"
        / accession
        / declared_identity
        / _cache_filename(accession, url)
    )
    if destination.is_file():
        return _verify_file(
            destination,
            expected_size=size_bytes,
            expected_md5=md5,
            expected_sha256=sha256,
            label=accession,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    if partial.is_file() and partial.stat().st_size > size_bytes:
        partial.unlink()

    last_error: OSError | None = None
    for _attempt in range(8):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == size_bytes:
            break
        headers = {"User-Agent": "organelleverse-himt-release-fixture/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            url,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                status = getattr(response, "status", None)
                if offset:
                    content_range = response.headers.get("Content-Range")
                    expected_prefix = f"bytes {offset}-"
                    expected_suffix = f"/{size_bytes}"
                    if (
                        status != 206
                        or content_range is None
                        or not content_range.startswith(expected_prefix)
                        or not content_range.endswith(expected_suffix)
                    ):
                        raise OrganelleDependencyError(
                            code="fixture.source_range_not_honored",
                            message=f"{accession} source did not honor the requested byte range",
                            details={
                                "offset": offset,
                                "status": status,
                                "content_range": content_range,
                            },
                        )
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
        except OSError as error:
            last_error = error
            continue

        downloaded_size = partial.stat().st_size
        if downloaded_size > size_bytes:
            partial.unlink()
            raise OrganelleDependencyError(
                code="fixture.source_size_mismatch",
                message=f"{accession} byte size exceeds its source manifest",
                details={"expected": size_bytes, "actual": downloaded_size},
            )
        if downloaded_size == offset:
            break

    if not partial.is_file() or partial.stat().st_size != size_bytes:
        raise OrganelleDependencyError(
            code="fixture.source_download_incomplete",
            message=f"{accession} source download remained incomplete after resumable attempts",
            details={
                "expected": size_bytes,
                "actual": partial.stat().st_size if partial.is_file() else 0,
                "last_error": str(last_error) if last_error is not None else None,
            },
            retryable=True,
        )

    try:
        actual = _verify_file(
            partial,
            expected_size=size_bytes,
            expected_md5=md5,
            expected_sha256=sha256,
            label=accession,
        )
    except OrganelleDependencyError:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)
    return actual.model_copy(update={"path": destination})


def _source_identity(source: FixtureSource, local: _LocalFile) -> SourceIdentity:
    return SourceIdentity(
        order=source.order,
        accession=source.accession,
        url=source.url,
        size_bytes=local.size_bytes,
        md5=local.md5,
        sha256=local.sha256,
    )


def _reference_identity(reference: FixtureReference, local: _LocalFile) -> ReferenceIdentity:
    return ReferenceIdentity(
        accessions=reference.accessions,
        url=reference.url,
        size_bytes=local.size_bytes,
        md5=local.md5,
        sha256=local.sha256,
    )


def _logical_argv(
    profile: ProfileFixture,
    executable_name: str,
) -> tuple[tuple[str, ...], ...]:
    if profile.selection is None:
        return ()
    return tuple(
        (
            executable_name,
            "-x",
            profile.selection.preset,
            "-t",
            str(profile.selection.threads),
            "<reference.fasta>",
            f"<source:{source.accession}>",
            "-o",
            f"<paf:{source.accession}>",
        )
        for source in profile.sources
    )


def _recipe_identity(
    profile: ProfileFixture,
    sources: tuple[SourceIdentity, ...],
    reference: ReferenceIdentity | None,
    *,
    minimap2_version: str,
    minimap2_argv: tuple[tuple[str, ...], ...],
) -> str:
    payload = {
        "profile": profile.profile,
        "technology": profile.technology,
        "quality_state": profile.quality_state,
        "sources": [source.model_dump(mode="json") for source in sources],
        "reference": None if reference is None else reference.model_dump(mode="json"),
        "selection": (
            None if profile.selection is None else profile.selection.model_dump(mode="json")
        ),
        "minimap2_version": minimap2_version,
        "minimap2_argv": [list(argv) for argv in minimap2_argv],
        "output_format": "fasta",
        "fixture_recipe_version": 1,
    }
    if profile.selection is not None:
        payload["record_identity"] = "accession_plus_sha256_full_fastq_description"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _open_reads(path: Path):
    if _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _locate_minimap2(cache_root: Path) -> Path:
    override = os.environ.get("ORGANELLEVERSE_MINIMAP2_EXECUTABLE")
    if override:
        return Path(override)
    environment = EnvironmentManager(cache_root=cache_root).prepare(
        HIMT_ENVIRONMENT,
        policy="ensure",
    )
    return environment.require_executable("minimap2")


def _probe_minimap2_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as error:
        raise OrganelleDependencyError(
            code="fixture.minimap2_unavailable",
            message="could not start minimap2 for fixture preparation",
            details={"error": str(error)},
        ) from error
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not version:
        raise OrganelleDependencyError(
            code="fixture.minimap2_unavailable",
            message="minimap2 version probe failed",
            details={"returncode": completed.returncode},
        )
    return version


def _run_minimap2(argv: tuple[str, ...]) -> None:
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as error:
        raise OrganelleDependencyError(
            code="fixture.minimap2_failed",
            message="could not start minimap2 for fixture derivation",
            details={"error": str(error)},
        ) from error
    if completed.returncode != 0:
        raise OrganelleDependencyError(
            code="fixture.minimap2_failed",
            message="minimap2 failed while deriving a HiMT fixture",
            details={
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            },
        )


def _parse_paf(path: Path, recipe: SelectionRecipe) -> set[str]:
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            try:
                query_start = int(fields[2])
                query_end = int(fields[3])
                mapq = int(fields[11])
            except (IndexError, ValueError) as error:
                raise OrganelleDependencyError(
                    code="fixture.invalid_paf",
                    message="minimap2 emitted malformed PAF",
                    details={"line": line_number},
                ) from error
            if mapq >= recipe.min_mapq and query_end - query_start >= recipe.min_aligned_span:
                selected.add(fields[0])
    return selected


def _count_fasta(path: Path) -> int:
    records = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _record in SeqIO.parse(handle, "fasta"):
                records += 1
    except ValueError as error:
        raise OrganelleDependencyError(
            code="fixture.invalid_fasta",
            message="HiFi fixture is not valid FASTA",
            details={"error": str(error)},
        ) from error
    if records == 0:
        raise OrganelleDependencyError(
            code="fixture.invalid_fasta",
            message="HiFi fixture contains no FASTA records",
        )
    return records


def _manifest_to_fixture(
    manifest: _DerivationManifest,
    *,
    path: Path,
    profile: ProfileFixture,
) -> PreparedFixture:
    return PreparedFixture(
        profile=manifest.profile,
        technology=profile.technology,
        quality_state=profile.quality_state,
        path=path,
        sha256=manifest.sha256,
        size_bytes=manifest.size_bytes,
        records=manifest.records,
        source_identities=manifest.source_identities,
        reference_identity=manifest.reference_identity,
        recipe_identity=manifest.recipe_identity,
        minimap2_version=manifest.minimap2_version,
        minimap2_argv=manifest.minimap2_argv,
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if isinstance(value, BaseModel):
            content = value.model_dump_json(indent=2) + "\n"
        else:
            content = json.dumps(value, indent=2, sort_keys=True) + "\n"
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _derive_long_reads(
    profile: ProfileFixture,
    cache_root: Path,
    source_files: tuple[_LocalFile, ...],
    source_identities: tuple[SourceIdentity, ...],
    reference_file: _LocalFile,
    reference_identity: ReferenceIdentity,
    executable: Path,
    version: str,
) -> PreparedFixture:
    assert profile.selection is not None
    logical_argv = _logical_argv(profile, executable.name)
    recipe_identity = _recipe_identity(
        profile,
        source_identities,
        reference_identity,
        minimap2_version=version,
        minimap2_argv=logical_argv,
    )
    profile_root = cache_root / "fixtures" / "himt" / profile.profile
    recipe_dir = profile_root / recipe_identity
    output_path = recipe_dir / "derived.fasta"
    manifest_path = recipe_dir / "derivation_manifest.json"
    if output_path.is_file() and manifest_path.is_file():
        manifest = _DerivationManifest.model_validate_json(manifest_path.read_bytes())
        _verify_file(
            output_path,
            expected_size=manifest.size_bytes,
            expected_md5=None,
            expected_sha256=manifest.sha256,
            label=f"derived {profile.profile} fixture",
        )
        if manifest.recipe_identity != recipe_identity:
            raise OrganelleDependencyError(
                code="fixture.recipe_identity_mismatch",
                message="cached fixture recipe identity does not match its path",
            )
        return _manifest_to_fixture(manifest, path=output_path, profile=profile)

    profile_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".derive-", dir=profile_root))
    try:
        selected_by_source: list[set[str]] = []
        for source, local in zip(profile.sources, source_files, strict=True):
            paf_path = temporary_dir / f"{source.accession}.paf"
            actual_argv = (
                str(executable),
                "-x",
                profile.selection.preset,
                "-t",
                str(profile.selection.threads),
                str(reference_file.path),
                str(local.path),
                "-o",
                str(paf_path),
            )
            _run_minimap2(actual_argv)
            selected_by_source.append(_parse_paf(paf_path, profile.selection))

        temporary_output = temporary_dir / "derived.fasta"
        records_written = 0
        with temporary_output.open("w", encoding="utf-8", newline="\n") as output:
            for source, local, selected in zip(
                profile.sources,
                source_files,
                selected_by_source,
                strict=True,
            ):
                emitted: set[str] = set()
                try:
                    with _open_reads(local.path) as reads:
                        for record in SeqIO.parse(reads, "fastq"):
                            if record.id not in selected:
                                continue
                            original_identity = hashlib.sha256(
                                str(record.description).encode("utf-8")
                            ).hexdigest()
                            rewritten = profile.selection.id_format.format(
                                accession=source.accession,
                                original_id=original_identity,
                            )
                            if rewritten in emitted:
                                raise OrganelleDependencyError(
                                    code="fixture.duplicate_read_id",
                                    message=(
                                        "duplicate selected full FASTQ description in "
                                        f"{source.accession}"
                                    ),
                                    details={"read_id": rewritten},
                                )
                            emitted.add(rewritten)
                            record.id = rewritten
                            record.name = rewritten
                            record.description = ""
                            SeqIO.write(record, output, "fasta")
                            records_written += 1
                except OrganelleDependencyError:
                    raise
                except ValueError as error:
                    raise OrganelleDependencyError(
                        code="fixture.invalid_fastq",
                        message=f"could not parse {source.accession} FASTQ",
                        details={"error": str(error)},
                    ) from error

        if records_written == 0:
            raise OrganelleDependencyError(
                code="fixture.empty_derived_fasta",
                message="minimap2 selection produced no reads",
            )
        output_identity = _hash_file(temporary_output)
        manifest = _DerivationManifest(
            profile=profile.profile,
            sha256=output_identity.sha256,
            size_bytes=output_identity.size_bytes,
            records=records_written,
            source_identities=source_identities,
            reference_identity=reference_identity,
            recipe_identity=recipe_identity,
            minimap2_version=version,
            minimap2_argv=logical_argv,
            output_filename="derived.fasta",
        )
        _write_json_atomic(temporary_dir / "derivation_manifest.json", manifest)
        try:
            os.replace(temporary_dir, recipe_dir)
        except FileExistsError:
            shutil.rmtree(temporary_dir)
        return _manifest_to_fixture(manifest, path=output_path, profile=profile)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _prepare_hifi(
    profile: ProfileFixture,
    cache_root: Path,
    source_file: _LocalFile,
    source_identity: SourceIdentity,
) -> PreparedFixture:
    records = _count_fasta(source_file.path)
    expected_records = profile.sources[0].records
    if expected_records is not None and records != expected_records:
        raise OrganelleDependencyError(
            code="fixture.hifi_record_count_mismatch",
            message="HiFi FASTA record count does not match its source manifest",
            details={"expected": expected_records, "actual": records},
        )
    recipe_identity = _recipe_identity(
        profile,
        (source_identity,),
        None,
        minimap2_version="",
        minimap2_argv=(),
    )
    manifest = _DerivationManifest(
        profile="hifi",
        sha256=source_file.sha256,
        size_bytes=source_file.size_bytes,
        records=records,
        source_identities=(source_identity,),
        reference_identity=None,
        recipe_identity=recipe_identity,
        minimap2_version="",
        minimap2_argv=(),
        output_filename=source_file.path.name,
    )
    manifest_path = (
        cache_root / "fixtures" / "himt" / "hifi" / recipe_identity / "derivation_manifest.json"
    )
    if not manifest_path.is_file():
        _write_json_atomic(manifest_path, manifest)
    return _manifest_to_fixture(manifest, path=source_file.path, profile=profile)


def _write_expected(fixture: PreparedFixture, expected_dir: Path) -> None:
    _write_json_atomic(
        expected_dir / f"{fixture.profile}.expected.json",
        {
            "profile": fixture.profile,
            "sha256": fixture.sha256,
            "size_bytes": fixture.size_bytes,
            "records": fixture.records,
            "recipe_identity": fixture.recipe_identity,
        },
    )


def prepare_himt_fixture(
    profile: FixtureProfile,
    cache_root: Path | str,
    *,
    write_expected: bool = False,
    _manifest: FixtureManifest | None = None,
    _minimap2_executable: Path | str | None = None,
    _minimap2_version: str | None = None,
    _expected_dir: Path | None = None,
) -> PreparedFixture:
    """Prepare one verified, content-addressed real-data HiMT fixture."""
    cache = Path(cache_root).expanduser().resolve()
    fixture_manifest = _manifest or load_fixture_manifest()
    profile_spec = _profile_from_manifest(fixture_manifest, profile)

    source_files = tuple(
        _download_verified(
            cache_root=cache,
            accession=source.accession,
            url=source.url,
            size_bytes=source.size_bytes,
            md5=source.md5,
            sha256=source.sha256,
        )
        for source in profile_spec.sources
    )
    source_identities = tuple(
        _source_identity(source, local)
        for source, local in zip(profile_spec.sources, source_files, strict=True)
    )

    if profile == "hifi":
        prepared = _prepare_hifi(profile_spec, cache, source_files[0], source_identities[0])
    else:
        assert profile_spec.reference is not None
        reference_accession = "_".join(profile_spec.reference.accessions)
        reference_file = _download_verified(
            cache_root=cache,
            accession=reference_accession,
            url=profile_spec.reference.url,
            size_bytes=profile_spec.reference.size_bytes,
            md5=profile_spec.reference.md5,
            sha256=profile_spec.reference.sha256,
        )
        executable = (
            Path(_minimap2_executable)
            if _minimap2_executable is not None
            else _locate_minimap2(cache)
        )
        version = _minimap2_version or _probe_minimap2_version(executable)
        prepared = _derive_long_reads(
            profile_spec,
            cache,
            source_files,
            source_identities,
            reference_file,
            _reference_identity(profile_spec.reference, reference_file),
            executable,
            version,
        )

    if write_expected:
        _write_expected(prepared, _expected_dir or _expected_directory())
    return prepared


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=("hifi", "clr", "ont"),
        required=True,
    )
    parser.add_argument("--write-expected", action="store_true")
    arguments = parser.parse_args(argv)
    cache_root = Path(
        os.environ.get(
            "ORGANELLEVERSE_CACHE_ROOT",
            Path.home() / ".cache" / "organelleverse",
        )
    )
    for raw_profile in arguments.profile:
        profile: FixtureProfile = raw_profile
        prepared = prepare_himt_fixture(
            profile,
            cache_root,
            write_expected=arguments.write_expected,
        )
        print(
            json.dumps(
                {
                    "profile": profile,
                    "source_sha256": [item.sha256 for item in prepared.source_identities],
                    "recipe_identity": prepared.recipe_identity,
                    "derived_sha256": prepared.sha256,
                    "size_bytes": prepared.size_bytes,
                    "records": prepared.records,
                    "path": str(prepared.path),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
