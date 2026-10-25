"""Adapter that exposes GeneCode deterministic biology tools to OrganelleVerse.

GeneCode (https://github.com/zjuphD/GeneCode) is a local-first molecular
sequence and cloning workbench.  This module imports its ``server_pkg``
implementation at runtime from a configurable install path so its biology
functions can be reused as OrganelleVerse desktop tools without vendoring the
whole repository into ``site-packages``.

The default search order for the GeneCode installation is:

1. ``ORGANELLEVERSE_GENECODE_PATH`` environment variable.
2. ``<repo-root>/runtime/genecode`` (checked-out sidecar location).
3. ``/tmp/GeneCode`` (development fallback from an earlier clone).

When no installation is found, every call raises :exc:`GeneCodeNotFoundError`
with a message telling the user how to clone it.
"""

from __future__ import annotations

from organelleverse._layout import checkout_root

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Module-level cache so repeated client instances reuse one loaded GeneCode.
_LOADED_MODULES: dict[str, Any] = {}


def _default_search_paths() -> list[Path]:
    repo_root = checkout_root()
    paths: list[Path] = []
    env = os.environ.get("ORGANELLEVERSE_GENECODE_PATH", "").strip()
    if env:
        paths.append(Path(env).expanduser().resolve())
    paths.append(repo_root / "runtime" / "genecode")
    paths.append(Path("/tmp/GeneCode").resolve())
    return paths


class GeneCodeNotFoundError(RuntimeError):
    """Raised when GeneCode cannot be located on disk."""

    def __init__(self, searched: list[Path]) -> None:
        self.searched = searched
        locations = "\n  ".join(str(p) for p in searched)
        super().__init__(
            "GeneCode installation not found. Searched:\n  "
            f"{locations}\n\n"
            "Clone it with:\n"
            "  git clone https://github.com/zjuphD/GeneCode.git "
            "<path>\n\n"
            "Then either set ORGANELLEVERSE_GENECODE_PATH=<path> or place it at "
            "<repo-root>/runtime/genecode."
        )


class GeneCodeApiError(RuntimeError):
    """Raised when GeneCode's own API rejects a request."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class GeneCodeClient:
    """Lightweight wrapper around a GeneCode ``server_pkg`` installation."""

    def __init__(self, search_paths: list[Path] | None = None) -> None:
        self._search_paths = search_paths if search_paths is not None else _default_search_paths()
        self._root: Path | None = None
        self._bio: Any | None = None
        self._schemas: Any | None = None

    def _find_root(self) -> Path:
        if self._root is not None:
            return self._root
        for path in self._search_paths:
            candidate = path / "server_pkg" / "bio.py"
            if candidate.exists():
                self._root = path
                return path
        raise GeneCodeNotFoundError(self._search_paths)

    def _load(self) -> None:
        if self._bio is not None:
            return
        root = self._find_root()
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        bio = _LOADED_MODULES.get("bio")
        schemas = _LOADED_MODULES.get("schemas")
        if bio is None:
            import server_pkg.bio as bio_module
            import server_pkg.schemas as schemas_module

            bio = _LOADED_MODULES["bio"] = bio_module
            schemas = _LOADED_MODULES["schemas"] = schemas_module
        self._bio = bio
        self._schemas = schemas

    def _call(self, fn_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Load GeneCode lazily, then invoke ``fn_name`` on the bio module."""
        self._load()
        fn: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self._bio, fn_name)
        try:
            return fn(payload)
        except Exception as exc:
            api_err = getattr(self._schemas, "ApiError", None)
            if api_err is not None and isinstance(exc, api_err):
                raise GeneCodeApiError(str(exc)) from exc
            raise

    def parse_sequence(self, text: str, *, name: str | None = None) -> dict[str, Any]:
        """Parse a FASTA/GenBank/plain sequence string and return metadata."""
        payload: dict[str, Any] = {"text": text}
        if name:
            payload["name"] = name
        return self._call("parse_sequence_response", payload)

    def export_sequence(self, document: dict[str, Any], *, fmt: str = "fasta") -> dict[str, Any]:
        """Serialize a parsed document back to FASTA or GenBank text."""
        return self._call("export_sequence_response", {"document": document, "format": fmt})

    def scan_restriction_sites(
        self,
        sequence: str,
        *,
        vector_sequence: str | None = None,
        enzymes: list[str] | None = None,
        topology: str = "linear",
        vector_topology: str = "circular",
        label: str = "Restriction Scan",
    ) -> dict[str, Any]:
        """Scan insert (and optionally vector) for restriction enzyme sites."""
        payload: dict[str, Any] = {
            "sequence": sequence,
            "topology": topology,
            "label": label,
        }
        if vector_sequence is not None:
            payload["vectorSequence"] = vector_sequence
            payload["vectorTopology"] = vector_topology
        if enzymes:
            payload["enzymes"] = enzymes
        return self._call("scan_restriction_sites_response", payload)

    def design_rtqpcr(
        self,
        sequence: str,
        *,
        label: str = "RT-qPCR",
        species: str = "Homo sapiens",
        gene_symbol: str | None = None,
        accession: str | None = None,
        gdna_check: bool = True,
        include_probe: bool = False,
    ) -> dict[str, Any]:
        """Design RT-qPCR primers (and optional probe) for a target sequence."""
        payload: dict[str, Any] = {
            "label": label,
            "sequence": sequence,
            "species": species,
            "gdnaCheck": gdna_check,
            "includeProbe": include_probe,
        }
        if gene_symbol is not None:
            payload["geneSymbol"] = gene_symbol
        if accession is not None:
            payload["accession"] = accession
        return self._call("design_rt_response", payload)

    def design_cloning(
        self,
        fragments: list[str],
        *,
        method: str = "gibson",
        label: str = "Molecular Cloning",
        vector_sequence: str | None = None,
        vector_topology: str = "circular",
        left_homology: str | None = None,
        right_homology: str | None = None,
        homology_length: int = 25,
        type_iis_enzyme: str = "BsaI",
        golden_gate_clamp_length: int = 4,
    ) -> dict[str, Any]:
        """Design cloning primers for Gibson or Golden Gate assembly.

        GeneCode's payload contract: a single insert goes under ``sequence``,
        while multi-fragment assembly uses the ``fragments`` list (>= 2 items).
        """
        payload: dict[str, Any] = {
            "label": label,
            "method": method,
        }
        if len(fragments) >= 2:
            payload["fragments"] = fragments
        elif fragments:
            payload["sequence"] = fragments[0]
        if vector_sequence is not None:
            payload["vectorSequence"] = vector_sequence
            payload["vectorTopology"] = vector_topology
        if left_homology is not None:
            payload["leftHomology"] = left_homology
        if right_homology is not None:
            payload["rightHomology"] = right_homology
        if method == "gibson":
            payload["homologyLength"] = homology_length
        elif method == "golden_gate":
            payload["typeIisEnzyme"] = type_iis_enzyme
            payload["clampLength"] = golden_gate_clamp_length
        return self._call("design_cloning_response", payload)

    def design_sgrna(
        self,
        sequence: str,
        *,
        mode: str = "ko",
        pam_set: str = "spcas9_ngg",
        label: str = "sgRNA",
    ) -> dict[str, Any]:
        """Design sgRNAs for a target sequence using a chosen PAM set."""
        payload: dict[str, Any] = {
            "label": label,
            "sequence": sequence,
            "mode": mode,
            "pamSet": pam_set,
        }
        return self._call("design_sgrna_response", payload)
