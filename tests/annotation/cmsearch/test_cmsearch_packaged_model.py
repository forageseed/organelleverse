from __future__ import annotations

import pytest

from organelleverse.annotation.cmsearch.model import parse_cm
from tests._paths import PROJECT_ROOT

CM = PROJECT_ROOT / "src/organelleverse/annotation/data/models/trna/trna.cm"
pytestmark = pytest.mark.skipif(not CM.exists(), reason="packaged CM not built yet")


def test_packaged_trna_cm_parses():
    cm = parse_cm(CM)
    assert cm.clen > 50
    assert any(s.type == "MP" for s in cm.states)
    assert len(cm.states) > 100
