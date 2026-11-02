"""method="auto" assembles plant data with ovasm when it can; everything else keeps its route."""

from pathlib import Path

import pytest

from organelleverse import _ovasm
from organelleverse.assembly import LongReadLibrary, ShortReadLibrary
from organelleverse.assembly import routing as routing_module
from organelleverse.assembly.contracts import AssemblyRequest
from organelleverse.assembly.routing import (
    AUTO_ASSEMBLER_ENV,
    OVASM_AUTO_RULE_ID,
    prepare_routing_context,
    resolve_backend,
)
from organelleverse.io import read_reads


def _fastq(path: Path) -> Path:
    path.write_text("@r\nACGT\n+\n!!!!\n")
    return path


def _hifi(tmp_path: Path, name: str = "hifi.fastq"):
    return read_reads(
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi", quality_state="ccs", reads=_fastq(tmp_path / name)
            ),
        )
    )


def _route(data, organelle="mitochondrion", **request_options):
    request = AssemblyRequest(data=data, organelle=organelle, method="auto", **request_options)
    context = prepare_routing_context(request)
    return resolve_backend(data, organelle=organelle, method="auto", context=context), context


@pytest.fixture
def ovasm_available(monkeypatch):
    monkeypatch.setattr(_ovasm, "resolve_ovasm", lambda: "/opt/ovasm")
    monkeypatch.delenv(AUTO_ASSEMBLER_ENV, raising=False)


def test_plant_hifi_mitochondrion_goes_to_ovasm_without_a_genome_size(tmp_path, ovasm_available):
    route, context = _route(_hifi(tmp_path))
    assert route.selected_backend == "ovasm"
    assert route.rule_id == OVASM_AUTO_RULE_ID
    assert context.ovasm_default and context.threshold_multiplier is None


def test_plastid_goes_to_ovasm_too(tmp_path, ovasm_available):
    route, _ = _route(_hifi(tmp_path), organelle="plastid")
    assert (route.selected_backend, route.rule_id) == ("ovasm", OVASM_AUTO_RULE_ID)


def test_without_the_binary_the_old_route_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(_ovasm, "resolve_ovasm", lambda: None)
    monkeypatch.delenv(AUTO_ASSEMBLER_ENV, raising=False)
    route, context = _route(_hifi(tmp_path), organelle="plastid")
    assert route.selected_backend == "oatk" and not context.ovasm_default


def test_the_legacy_switch_restores_the_old_route(tmp_path, ovasm_available, monkeypatch):
    monkeypatch.setenv(AUTO_ASSEMBLER_ENV, "legacy")
    route, _ = _route(_hifi(tmp_path), organelle="plastid")
    assert route.selected_backend == "oatk"


def test_non_plant_data_keeps_its_route(tmp_path, ovasm_available):
    route, _ = _route(_hifi(tmp_path), organelle="plastid", taxon_group="animal")
    assert route.selected_backend != "ovasm"


def test_two_read_libraries_keep_their_route(tmp_path, ovasm_available):
    data = read_reads(
        short_libraries=(
            ShortReadLibrary(
                technology="illumina",
                layout="paired_end",
                read1=_fastq(tmp_path / "R1.fastq"),
                read2=_fastq(tmp_path / "R2.fastq"),
                read_length=150,
                insert_size=350,
            ),
        ),
        long_libraries=(
            LongReadLibrary(
                technology="pacbio_hifi", quality_state="ccs", reads=_fastq(tmp_path / "h.fastq")
            ),
        ),
    )
    route, context = _route(data, organelle="plastid")
    assert not context.ovasm_default and route.selected_backend != "ovasm"


def test_an_environment_hint_or_pinned_version_keeps_the_old_route(tmp_path, ovasm_available):
    _, context = _route(_hifi(tmp_path), organelle="plastid", backend_version="latest")
    assert not context.ovasm_default


def test_explicit_methods_are_untouched(tmp_path, ovasm_available):
    data = _hifi(tmp_path)
    request = AssemblyRequest(data=data, organelle="plastid", method="oatk")
    route = resolve_backend(
        data, organelle="plastid", method="oatk", context=prepare_routing_context(request)
    )
    assert (route.selected_backend, route.rule_id) == ("oatk", "explicit.backend")
    assert routing_module.OVASM_AUTO_RULE_ID == "auto.plant_ovasm"
