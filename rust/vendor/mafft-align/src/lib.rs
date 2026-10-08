//! Pairwise and multi-sequence alignment algorithms for MAFFT.
//!
//! Phase 4a: Single-sequence pairwise alignment (Galign11, Lalign11, genalign11).
//! Phase 5: Profile alignment (MSalignmm), FFT-accelerated alignment (Falign),
//!          local homology constraint building (pairlocalalign),
//!          and constrained alignment (Falign_localhom, partSalignmm).

mod dp;
mod global;
mod local;
mod genaffine;
mod profile;
mod multimtx;
mod fft_align;
mod constraints;
mod constrained_align;
mod msalign;

pub use dp::{Alignment, AlignOp, GapModel, matrix_i32_to_f64};
pub use global::global_align;
pub use local::{local_align, LocalAlignment};
pub use genaffine::{genaffine_local_align, GenAffineGapModel};
pub use profile::{
    Profile, profile_align, profile_align_imp, profile_align_imp_with_tiebreak,
    profile_align_imp_with_boundary, profile_align_imp_multimtx, BoundaryFreqs,
    pairwise_align11, pairwise_align11_ex, align_with_anchors,
    reset_dp_pools, reset_cpmx_memo,
};
pub use multimtx::MultiMtx;
pub use fft_align::{fft_profile_align, find_fft_anchors, FftAlignParams, Anchor};
pub use constraints::{
    build_local_homology_table, build_homology_table,
    build_homology_table_with_unalign, build_imp_matrix,
    recompute_importance, PairAligner, FASTATHRESHOLD_DEFAULT,
    extract_putlocalhom2_regions, build_seed_homology_table,
    merge_homology_tables, parse_hat3_seed, SeedGroup,
};
pub use constrained_align::{
    constrained_profile_align, partial_profile_align, ConstrainedAlignParams,
};
pub use msalign::msalignmm;
