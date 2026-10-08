from pathlib import Path

import pytest

from organelleverse.core.errors import OrganelleExecutionError
from organelleverse.pangenome.tuning_service import _require_sample_pairs


def test_mash_pairs_reject_pooled_self_distance_and_duplicates():
    files = [Path('/sample files/a.fa'), Path('/sample files/b.fa')]
    with pytest.raises(OrganelleExecutionError, match='every biological-sample pair'):
        _require_sample_pairs('all.msh\tall.msh\t0\t0\t100/100\n', files)
    valid = ''.join(f'{a}\t{b}\t0.01\t0\t100/1000\n' for a in files for b in files)
    _require_sample_pairs(valid, files)
    with pytest.raises(OrganelleExecutionError, match='every biological-sample pair'):
        _require_sample_pairs(valid + valid.splitlines()[0] + '\n', files)
