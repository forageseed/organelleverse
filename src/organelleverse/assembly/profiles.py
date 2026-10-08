"""Side-effect-free assembly data profile identities."""

from __future__ import annotations

from enum import StrEnum


class AssemblyProfile(StrEnum):
    ILLUMINA_PE = "illumina_pe"
    ILLUMINA_SE = "illumina_se"
    PACBIO_HIFI = "pacbio_hifi"
    PACBIO_CLR_RAW = "pacbio_clr_raw"
    PACBIO_CLR_CORRECTED = "pacbio_clr_corrected"
    ONT_RAW = "ont_raw"
    ONT_CORRECTED = "ont_corrected"
    ONT_HQ = "ont_hq"
    ONT_DUPLEX = "ont_duplex"
    ONT_RAW_ILLUMINA = "ont_raw_illumina"
    PACBIO_CLR_RAW_ILLUMINA = "pacbio_clr_raw_illumina"


__all__ = ["AssemblyProfile"]
