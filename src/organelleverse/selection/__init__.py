"""Selection suite: Ka/Ks + CodeML preprocessing + four selection models.

Model classes:
  - Branch models    (branch_model)      — lineage-specific ω
  - Site models      (site_model)        — positive-selection sites (M0-M8)
  - Branch-site      (branch_site_model) — lineage-specific PS sites + BEB
  - Clade models     (clade_model)       — divergent selection (CmC)

HyPhy analyses (external ``hyphy`` >= 2.5, same codon alignment + foreground
labelling as the codeml models):
  - hyphy_busted  — gene-wide episodic diversifying selection
  - hyphy_absrel  — branches under episodic diversifying selection
  - hyphy_relax   — relaxed vs intensified selection on test branches
  - hyphy_meme / hyphy_fel — site-level episodic / pervasive selection

File-producing tools use platform-managed run directories for intermediates.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from ..core.result import OrganelleResult
from .kaks import kaks
from .codeml import (
    validate_cds,
    translate_cds,
    align_protein,
    pal2nal,
    to_paml,
    prepare_codeml,
)
from .models import (
    branch_model,
    site_model,
    branch_site_model,
    clade_model,
    batch_codeml,
    parse_codeml_output,
    likelihood_ratio_test,
)
from .codeml_core import compute_kaks, validate_cds_sequences
from .kaks_calculator import (
    kaks_calculator,
    check_kaks_calculator,
    parse_kaks_output,
    run_kaks_calculator,
    SUPPORTED_METHODS as KAKS_METHODS,
)
from .service import fasta_to_axt, run_codeml, run_kaks_calculator_workflow
from .hyphy import (
    check_hyphy,
    hyphy_absrel,
    hyphy_busted,
    hyphy_fel,
    hyphy_meme,
    hyphy_relax,
    parse_hyphy_json,
)


def write(obj: Any, output: str | Path, *, kind: str | None = None, **kwargs: Any) -> Any:
    """Write selection results or workflow inputs.

    Result objects go through the canonical publication boundary. Workflow
    preparations that need source files use ``kind=``: ``codeml_inputs``,
    ``branch_model``, ``site_model``, ``branch_site_model`` or ``clade_model``.
    """
    if isinstance(obj, OrganelleResult):
        from .writer import materialize_result

        return materialize_result(obj, output)
    if kind in {None, "codeml_inputs"}:
        from .codeml import write_codeml_inputs

        return write_codeml_inputs(obj, output_dir=output, **kwargs)
    if kind == "branch_model":
        from .models import write_branch_model

        return write_branch_model(alignment=obj, output_dir=output, **kwargs)
    if kind == "site_model":
        from .models import write_site_model

        return write_site_model(alignment=obj, output_dir=output, **kwargs)
    if kind == "branch_site_model":
        from .models import write_branch_site_model

        return write_branch_site_model(alignment=obj, output_dir=output, **kwargs)
    if kind == "clade_model":
        from .models import write_clade_model

        return write_clade_model(alignment=obj, output_dir=output, **kwargs)
    raise ValueError(f"unknown selection write kind: {kind!r}")


save = write


__all__ = [
    # Ka/Ks (pure Python NG86)
    "kaks",
    "write",
    "save",
    # Ka/Ks (KaKs_Calculator 3.0 — all 10 methods)
    "kaks_calculator",
    "run_kaks_calculator_workflow",
    "check_kaks_calculator",
    "fasta_to_axt",
    "parse_kaks_output",
    "run_kaks_calculator",
    "KAKS_METHODS",
    # CodeML preprocessing
    "validate_cds",
    "translate_cds",
    "align_protein",
    "pal2nal",
    "to_paml",
    "prepare_codeml",
    # Four selection models
    "branch_model",
    "site_model",
    "branch_site_model",
    "clade_model",
    # Execution
    "run_codeml",
    "batch_codeml",
    "parse_codeml_output",
    "likelihood_ratio_test",
    # HyPhy (external hyphy >= 2.5)
    "check_hyphy",
    "hyphy_busted",
    "hyphy_absrel",
    "hyphy_relax",
    "hyphy_meme",
    "hyphy_fel",
    "parse_hyphy_json",
]
