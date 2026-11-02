"""A parsed reference read back from the on-disk store equals a fresh parse; a changed file is parsed again."""

import shutil
from pathlib import Path

from organelleverse.annotation.plastome import references as refs


def _signature(ref):
    return (
        ref.path,
        str(ref.record.seq),
        ref.kmers,
        tuple(
            (f.feature_type, f.gene, f.sequence, f.exon_index, f.exon_count, str(f.feature.location),
             sorted((k, tuple(v)) for k, v in f.feature.qualifiers.items()))
            for f in ref.features
        ),
    )


def test_stored_parse_equals_fresh_parse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    data = Path(refs.__file__).resolve().parents[1] / "data/plastome/references"
    for gb in sorted(data.glob("*.gb"))[:3]:
        fresh = refs._parse_reference(gb)
        first = refs._parse_reference_stored(gb)  # parses and stores
        assert list((tmp_path / "cache/plastome-references").glob("*.pickle"))
        second = refs._parse_reference_stored(gb)  # read back
        assert _signature(first) == _signature(fresh) == _signature(second)


def test_changed_file_is_parsed_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORGANELLEVERSE_CACHE_ROOT", str(tmp_path / "cache"))
    data = Path(refs.__file__).resolve().parents[1] / "data/plastome/references"
    src = sorted(data.glob("*.gb"))[0]
    gb = tmp_path / src.name
    shutil.copy(src, gb)
    before = refs._parse_reference_stored(gb)
    text = gb.read_text()
    gb.write_text(text.replace("/gene=", "/gene=x", 1))
    after = refs._parse_reference_stored(gb)
    assert _signature(after) == _signature(refs._parse_reference(gb))
    assert _signature(after) != _signature(before)
    assert len(list((tmp_path / "cache/plastome-references").glob("*.pickle"))) == 2
