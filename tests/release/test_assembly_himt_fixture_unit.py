"""Hermetic tests for deterministic HiMT release-fixture provisioning."""

from __future__ import annotations

import gzip
import hashlib
import http.server
import io
import json
import socketserver
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from Bio import SeqIO
from pydantic import ValidationError

from organelleverse.core.errors import OrganelleDependencyError
from tests.release.himt_fixture import (
    FixtureManifest,
    FixtureReference,
    FixtureSource,
    ProfileFixture,
    SelectionRecipe,
    _download_verified,
    load_fixture_manifest,
    prepare_himt_fixture,
)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def file_server(tmp_path: Path) -> Iterator[str]:
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args, directory=str(tmp_path), **kwargs
    )
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _write_fastq_gz(path: Path, records: list[tuple[str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record_id, sequence in records:
            handle.write(f"@{record_id}\n{sequence}\n+\n{'I' * len(sequence)}\n")


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n{sequence}\n")


def _source(
    path: Path,
    *,
    order: int,
    accession: str,
    url: str,
    records: int | None,
) -> FixtureSource:
    content = path.read_bytes()
    return FixtureSource(
        order=order,
        accession=accession,
        url=url,
        size_bytes=len(content),
        md5=hashlib.md5(content).hexdigest(),
        sha256=hashlib.sha256(content).hexdigest(),
        records=records,
        platform="test-platform",
        library_strategy="WGS",
    )


def _profile(
    tmp_path: Path,
    file_server: str,
    *,
    profile: str = "clr",
    technology: str = "pacbio_clr",
    quality_state: str = "raw",
    source_records: tuple[tuple[tuple[str, str], ...], ...] = (
        (("read_a1", "ACGT" * 50), ("read_a2", "TGCA" * 50)),
        (("read_b1", "AACC" * 50),),
    ),
    min_mapq: int = 5,
) -> ProfileFixture:
    sources: list[FixtureSource] = []
    for index, records in enumerate(source_records, start=1):
        accession = f"RUN{index}"
        source_path = tmp_path / f"{accession}.fastq.gz"
        _write_fastq_gz(source_path, list(records))
        sources.append(
            _source(
                source_path,
                order=index - 1,
                accession=accession,
                url=f"{file_server}/{source_path.name}",
                records=len(records),
            )
        )

    reference_path = tmp_path / "reference.fasta"
    _write_fasta(reference_path, [("mito", "ACGT" * 1000), ("plastid", "TGCA" * 1000)])
    reference_bytes = reference_path.read_bytes()
    reference = FixtureReference(
        accessions=("MITO.1", "PLASTID.1"),
        url=f"{file_server}/{reference_path.name}",
        size_bytes=len(reference_bytes),
        md5=hashlib.md5(reference_bytes).hexdigest(),
        sha256=hashlib.sha256(reference_bytes).hexdigest(),
    )
    preset = "map-ont" if profile == "ont" else "map-pb"
    return ProfileFixture(
        profile=cast(Any, profile),
        technology=technology,
        quality_state=quality_state,
        biosample="TEST-SAMPLE",
        study="TEST-STUDY",
        platform="test-platform",
        license="test-only",
        sources=tuple(sources),
        reference=reference,
        selection=SelectionRecipe(
            mapper="minimap2",
            preset=preset,
            threads=1,
            min_mapq=min_mapq,
            min_aligned_span=10,
            id_format="{accession}|{original_id}",
        ),
    )


def _hifi_profile(tmp_path: Path, file_server: str) -> ProfileFixture:
    path = tmp_path / "demo.fa"
    _write_fasta(path, [("hifi1", "ACGT" * 100), ("hifi2", "TGCA" * 100)])
    return ProfileFixture(
        profile="hifi",
        technology="pacbio_hifi",
        quality_state="ccs",
        biosample="TEST-HIFI",
        study="TEST-HIFI-STUDY",
        platform="test-platform",
        license="test-only",
        sources=(
            _source(
                path,
                order=0,
                accession="demo",
                url=f"{file_server}/{path.name}",
                records=2,
            ),
        ),
        reference=None,
        selection=None,
    )


def _manifest(*profiles: ProfileFixture) -> FixtureManifest:
    return FixtureManifest(profiles=profiles)


def _replace_source(profile: ProfileFixture, index: int, **updates: object) -> ProfileFixture:
    sources = list(profile.sources)
    sources[index] = sources[index].model_copy(update=updates)
    return profile.model_copy(update={"sources": tuple(sources)})


def _fake_minimap2(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-bin" / "minimap2"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv[1:]:\n"
        "    print('2.24-r1122')\n"
        "    raise SystemExit(0)\n"
        "output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "source = Path(sys.argv[sys.argv.index('-o') - 1])\n"
        "accession = source.parent.parent.name\n"
        "fail = os.environ.get('OV_FAKE_MINIMAP2_FAIL')\n"
        "if fail and (fail == '*' or fail == accession):\n"
        "    print('forced failure', file=sys.stderr)\n"
        "    raise SystemExit(7)\n"
        "selected = json.loads(os.environ.get('OV_FAKE_MINIMAP2_SELECT', '{}'))\n"
        "lines = []\n"
        "for read_id in selected.get(accession, []):\n"
        "    lines.append('\\t'.join([read_id, '200', '0', '200', '+', 'ref', "
        "'10000', '0', '200', '200', '200', '60']))\n"
        "output.write_text(('\\n'.join(lines) + ('\\n' if lines else '')), encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _prepare(
    profile: str,
    cache_root: Path,
    manifest: FixtureManifest,
    executable: Path,
    *,
    write_expected: bool = False,
    expected_dir: Path | None = None,
):
    return prepare_himt_fixture(
        cast(Any, profile),
        cache_root,
        write_expected=write_expected,
        _manifest=manifest,
        _minimap2_executable=executable,
        _minimap2_version="2.24-r1122",
        _expected_dir=expected_dir,
    )


def test_manifest_models_reject_unknown_missing_and_out_of_order_fields(tmp_path: Path) -> None:
    valid = json.loads(
        (Path(__file__).parent / "fixtures" / "assembly" / "himt" / "sources.json").read_text(
            encoding="utf-8"
        )
    )
    unknown = json.loads(json.dumps(valid))
    unknown["extra"] = True
    missing = json.loads(json.dumps(valid))
    del missing["profiles"][0]["technology"]
    out_of_order = json.loads(json.dumps(valid))
    out_of_order["profiles"][1]["sources"][0]["order"] = 4

    for index, payload in enumerate((unknown, missing, out_of_order)):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_fixture_manifest(path)


def test_cache_miss_then_exact_hit_avoids_network_and_minimap2(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path, file_server)
    manifest = _manifest(profile)
    executable = _fake_minimap2(tmp_path)
    monkeypatch.setenv(
        "OV_FAKE_MINIMAP2_SELECT",
        json.dumps({"RUN1": ["read_a1"], "RUN2": ["read_b1"]}),
    )
    first = _prepare("clr", tmp_path / "cache", manifest, executable)
    assert first.records == 2
    assert first.path.is_file()

    monkeypatch.setenv("OV_FAKE_MINIMAP2_FAIL", "*")
    with patch(
        "tests.release.himt_fixture.urllib.request.urlopen",
        side_effect=AssertionError("cache hit attempted network access"),
    ):
        second = _prepare("clr", tmp_path / "cache", manifest, executable)
    assert second == first


def test_interrupted_source_download_resumes_with_verified_range(
    tmp_path: Path,
) -> None:
    content = b"verified-public-fixture" * 32
    split = len(content) // 2
    ranges: list[str | None] = []

    class Response(io.BytesIO):
        def __init__(self, body: bytes, *, status: int, content_range: str | None = None):
            super().__init__(body)
            self.status = status
            self.headers = {} if content_range is None else {"Content-Range": content_range}

    def open_interrupted_then_resumed(request: object, *, timeout: int) -> Response:
        assert timeout == 3600
        range_header = cast(Any, request).get_header("Range")
        ranges.append(range_header)
        if len(ranges) == 1:
            return Response(content[:split], status=200)
        assert range_header == f"bytes={split}-"
        return Response(
            content[split:],
            status=206,
            content_range=f"bytes {split}-{len(content) - 1}/{len(content)}",
        )

    with patch(
        "tests.release.himt_fixture.urllib.request.urlopen",
        side_effect=open_interrupted_then_resumed,
    ):
        downloaded = _download_verified(
            cache_root=tmp_path / "cache",
            accession="RUN1",
            url="https://example.invalid/RUN1.fastq.gz",
            size_bytes=len(content),
            md5=hashlib.md5(content).hexdigest(),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    assert ranges == [None, f"bytes={split}-"]
    assert downloaded.path.read_bytes() == content


@pytest.mark.parametrize(
    ("updates", "error_code"),
    [
        ({"size_bytes": 1}, "fixture.source_size_mismatch"),
        ({"md5": "0" * 32}, "fixture.source_md5_mismatch"),
        ({"sha256": "0" * 64}, "fixture.source_sha256_mismatch"),
    ],
)
def test_source_integrity_failures_are_specific(
    tmp_path: Path,
    file_server: str,
    updates: dict[str, object],
    error_code: str,
) -> None:
    profile = _replace_source(_profile(tmp_path, file_server), 0, **updates)
    with pytest.raises(OrganelleDependencyError) as raised:
        _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    assert raised.value.code == error_code


def test_changed_reference_fails_before_minimap2(tmp_path: Path, file_server: str) -> None:
    profile = _profile(tmp_path, file_server)
    assert profile.reference is not None
    profile = profile.model_copy(
        update={"reference": profile.reference.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(OrganelleDependencyError) as raised:
        _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    assert raised.value.code == "fixture.source_sha256_mismatch"


def test_nonzero_minimap2_is_a_typed_failure(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path, file_server)
    monkeypatch.setenv("OV_FAKE_MINIMAP2_FAIL", "RUN1")
    with pytest.raises(OrganelleDependencyError) as raised:
        _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    assert raised.value.code == "fixture.minimap2_failed"


def test_truncated_fastq_is_detected_while_streaming(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path, file_server, source_records=((("broken", "ACGT" * 50),),))
    source_path = tmp_path / "RUN1.fastq.gz"
    with gzip.open(source_path, "wt", encoding="utf-8") as handle:
        handle.write("@broken\nACGT\n+\n")
    profile = _replace_source(
        profile,
        0,
        size_bytes=source_path.stat().st_size,
        md5=hashlib.md5(source_path.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("OV_FAKE_MINIMAP2_SELECT", json.dumps({"RUN1": ["broken"]}))
    with pytest.raises(OrganelleDependencyError) as raised:
        _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    assert raised.value.code == "fixture.invalid_fastq"


def test_duplicate_selected_read_id_is_rejected(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (("duplicate", "ACGT" * 50), ("duplicate", "TGCA" * 50))
    profile = _profile(tmp_path, file_server, source_records=(records,))
    monkeypatch.setenv("OV_FAKE_MINIMAP2_SELECT", json.dumps({"RUN1": ["duplicate"]}))
    with pytest.raises(OrganelleDependencyError) as raised:
        _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    assert raised.value.code == "fixture.duplicate_read_id"


def test_repeated_paf_query_id_uses_unique_full_fastq_descriptions(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (("duplicate /1", "ACGT" * 50), ("duplicate /3", "TGCA" * 50))
    profile = _profile(tmp_path, file_server, source_records=(records,))
    monkeypatch.setenv("OV_FAKE_MINIMAP2_SELECT", json.dumps({"RUN1": ["duplicate"]}))

    fixture = _prepare(
        "clr",
        tmp_path / "cache",
        _manifest(profile),
        _fake_minimap2(tmp_path),
    )

    derived = list(SeqIO.parse(fixture.path, "fasta"))
    assert [record.id for record in derived] == [
        f"RUN1|{hashlib.sha256(b'duplicate /1').hexdigest()}",
        f"RUN1|{hashlib.sha256(b'duplicate /3').hexdigest()}",
    ]
    assert all(len(record.id) <= 100 for record in derived)
    assert [str(record.seq) for record in derived] == ["ACGT" * 50, "TGCA" * 50]


def test_extraction_preserves_source_then_record_order_and_sequence(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(tmp_path, file_server)
    monkeypatch.setenv(
        "OV_FAKE_MINIMAP2_SELECT",
        json.dumps({"RUN1": ["read_a2", "read_a1"], "RUN2": ["read_b1"]}),
    )
    fixture = _prepare("clr", tmp_path / "cache", _manifest(profile), _fake_minimap2(tmp_path))
    records = list(SeqIO.parse(fixture.path, "fasta"))
    assert [record.id for record in records] == [
        f"RUN1|{hashlib.sha256(b'read_a1').hexdigest()}",
        f"RUN1|{hashlib.sha256(b'read_a2').hexdigest()}",
        f"RUN2|{hashlib.sha256(b'read_b1').hexdigest()}",
    ]
    assert [str(record.seq) for record in records] == [
        "ACGT" * 50,
        "TGCA" * 50,
        "AACC" * 50,
    ]


def test_recipe_identity_is_machine_independent_and_changes_with_recipe(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_FAKE_MINIMAP2_SELECT", json.dumps({"RUN1": ["read_a1"]}))
    profile = _profile(tmp_path, file_server, source_records=((("read_a1", "ACGT" * 50),),))
    first = _prepare(
        "clr", tmp_path / "cache-a", _manifest(profile), _fake_minimap2(tmp_path / "host-a")
    )
    second = _prepare(
        "clr", tmp_path / "cache-b", _manifest(profile), _fake_minimap2(tmp_path / "host-b")
    )
    changed = _profile(
        tmp_path,
        file_server,
        source_records=((("read_a1", "ACGT" * 50),),),
        min_mapq=10,
    )
    third = _prepare(
        "clr", tmp_path / "cache-c", _manifest(changed), _fake_minimap2(tmp_path / "host-c")
    )
    assert first.recipe_identity == second.recipe_identity
    assert third.recipe_identity != first.recipe_identity
    assert str(tmp_path) not in first.recipe_identity


@pytest.mark.parametrize("profile_name", ["hifi", "clr", "ont"])
def test_every_profile_has_a_recipe_identity_and_writes_expected_manifest(
    tmp_path: Path,
    file_server: str,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
) -> None:
    if profile_name == "hifi":
        profile = _hifi_profile(tmp_path, file_server)
    else:
        record_id = f"{profile_name}1"
        sequence = "ACGT" * 50 if profile_name == "clr" else "TGCA" * 50
        profile = _profile(
            tmp_path,
            file_server,
            profile=profile_name,
            technology="pacbio_clr" if profile_name == "clr" else "ont",
            source_records=(((record_id, sequence),),),
        )
    monkeypatch.setenv(
        "OV_FAKE_MINIMAP2_SELECT",
        json.dumps({"RUN1": ["clr1", "ont1"]}),
    )
    expected_dir = tmp_path / "expected"
    fixture = _prepare(
        profile_name,
        tmp_path / f"cache-{profile_name}",
        _manifest(profile),
        _fake_minimap2(tmp_path / profile_name),
        write_expected=True,
        expected_dir=expected_dir,
    )
    assert fixture.recipe_identity.startswith("sha256:")
    assert len(fixture.recipe_identity) == 71
    expected = json.loads(
        (expected_dir / f"{profile_name}.expected.json").read_text(encoding="utf-8")
    )
    assert expected == {
        "profile": profile_name,
        "sha256": fixture.sha256,
        "size_bytes": fixture.size_bytes,
        "records": fixture.records,
        "recipe_identity": fixture.recipe_identity,
    }
