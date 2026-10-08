/// Profile-to-profile (group-to-group) alignment.
///
/// Ports the C `MSalignmm()` from MSalignmm.c.
///
/// Aligns two groups of sequences by computing position-specific frequency
/// matrices (profiles) and running affine-gap DP on the profile scores.

use std::cell::RefCell;

use crate::dp::{AlignOp, Alignment, GapModel};

// §C.2 path-(1) allocation arena: amortise the two big 2D DP matrices
// (`h` and `ijp`) across calls to `profile_align_imp_with_boundary`.
// Per `PROFILING.md`, ~9% of FFT-NS-i `--maxiterate 50` runtime is in
// `Vec` allocator churn inside the profile DP; pooling `h` (n×m f64,
// typically ~3 MB) and `ijp` (n×m i32, ~1.5 MB) is the cheapest lever
// because the per-cell DP body never reads stale data — every read
// cell is written by either the boundary init (line 631-632) or the
// DP body before any traceback read.
//
// Thread-local so rayon workers don't contend; each worker gets its
// own pool. Growth-only resize (matches C `A__align`'s `static TLS`
// buffer amortisation — see `MAFFT_UPSTREAM_REPORT.md` for the
// downstream effect on tied-cell traceback). RefCell because the DP
// body needs `&mut` to the rows.
/// Per-thread scratch buffers reused across `profile_align_imp_multimtx`
/// calls. Mirrors C `A__align`'s `static TLS` strategy — every field is
/// either fully overwritten before read on each call, or explicitly
/// reset by the function before use, so cross-call data carried in the
/// buffer never leaks into the alignment.
///
/// Profile (BB30004 post-const-generic-split, 2026-06-02) showed the
/// per-call vec allocations inside `profile_align_imp_multimtx` summing
/// to ~3–4 % of total runtime: `cpmx{1,2}_sparse` (nested
/// `Vec<Vec<(usize, f64)>>` allocations), four gap-cost Vecs of size
/// `n+1` / `m+1`, plus six per-row dense Vecs (`mj`, `mpj`, `currentw`,
/// `previousw`, `initverticalw`, `lastverticalw`), plus `scarr`
/// (allocated *per row* inside `match_calc_row`, so `~n` allocs per DP
/// call), plus six warp-state Vecs when `try_warp == true`. All of them
/// move here.
#[derive(Default)]
struct DpScratch {
    ogcp1: Vec<f64>,
    fgcp1: Vec<f64>,
    ogcp2: Vec<f64>,
    fgcp2: Vec<f64>,
    initverticalw: Vec<f64>,
    currentw: Vec<f64>,
    previousw: Vec<f64>,
    lastverticalw: Vec<f64>,
    mj: Vec<f64>,
    mpj: Vec<usize>,
    scarr: Vec<f64>,
    cpmx1_sparse: Vec<Vec<(usize, f64)>>,
    cpmx2_sparse: Vec<Vec<(usize, f64)>>,
    wmrecords: Vec<f64>,
    prevwmrecords: Vec<f64>,
    warpi: Vec<i32>,
    warpj: Vec<i32>,
    prevwarpi: Vec<i32>,
    prevwarpj: Vec<i32>,
    warpis: Vec<i32>,
    warpjs: Vec<i32>,
}

impl DpScratch {
    const fn new() -> Self {
        Self {
            ogcp1: Vec::new(),
            fgcp1: Vec::new(),
            ogcp2: Vec::new(),
            fgcp2: Vec::new(),
            initverticalw: Vec::new(),
            currentw: Vec::new(),
            previousw: Vec::new(),
            lastverticalw: Vec::new(),
            mj: Vec::new(),
            mpj: Vec::new(),
            scarr: Vec::new(),
            cpmx1_sparse: Vec::new(),
            cpmx2_sparse: Vec::new(),
            wmrecords: Vec::new(),
            prevwmrecords: Vec::new(),
            warpi: Vec::new(),
            warpj: Vec::new(),
            prevwarpi: Vec::new(),
            prevwarpj: Vec::new(),
            warpis: Vec::new(),
            warpjs: Vec::new(),
        }
    }
}

thread_local! {
    static DP_H_POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
    static DP_IJP_POOL: RefCell<Vec<Vec<i32>>> = const { RefCell::new(Vec::new()) };
    static DP_SCRATCH: RefCell<DpScratch> = const { RefCell::new(DpScratch::new()) };
    /// `--c-compat` opt-in: mirrors C's `static TLS` memoization in
    /// `Salignmm.c::A__align` (lines 1091-1094, 1446-1450, 2203-2206).
    /// When `Profile::from_aligned_with_memo` is called repeatedly with
    /// matching `(firstmem, lgth, icyc == previous_icyc + 1)`, the new
    /// profile is built via `cpmx_calc_add`-equivalent semantics
    /// (multiply existing freqs by orieff, add neweff for the new
    /// sequence) instead of `cpmx_calc_new` (zero + rebuild). The two
    /// paths produce mathematically equivalent results but differ in
    /// FP precision — and that 1-ULP drift biases C's tied-DP-cell
    /// selection. To reproduce C bit-for-bit on tied cells, we must
    /// replicate the same per-thread state machine.
    static CPMX_MEMO: RefCell<CpmxMemoState> = const { RefCell::new(CpmxMemoState::new()) };
}

/// Per-thread cpmx memoization state mirroring C `Salignmm.c` statics
/// `previousfirstlen` / `previousicyc` / `previousfirstmem` /
/// `previouscall` plus the static `cpmx1` buffer. Activated only when
/// the engine is built with `c_compat=true`.
struct CpmxMemoState {
    /// True if at least one prior `from_aligned_with_memo` ran on this
    /// thread. Mirrors C `previouscall` (set via `calledbyfulltreebase`
    /// in C; here we only ever set it when c_compat is on, so the
    /// presence-of-prior-call is the bit we need).
    previous_call: bool,
    previous_firstmem: i32,
    previous_icyc: usize,
    previous_first_len: usize,
    /// Persistent cpmx buffer: `freqs[pos][k]` for pos in
    /// `[0..previous_first_len)`, k in `[0..nalphabets)`. Sized to the
    /// largest `lgth` × `nalphabets` seen, grown but never shrunk.
    freqs: Vec<Vec<f64>>,
    gap_freq: Vec<f64>,
    opening: Vec<f64>,
    closing: Vec<f64>,
}

impl CpmxMemoState {
    const fn new() -> Self {
        Self {
            previous_call: false,
            previous_firstmem: -1,
            previous_icyc: 0,
            previous_first_len: 0,
            freqs: Vec::new(),
            gap_freq: Vec::new(),
            opening: Vec::new(),
            closing: Vec::new(),
        }
    }
}

/// Drop pooled DP buffers on this thread. The engine calls this between
/// retree passes when investigating cross-pass state leak; in production
/// it's a no-op for correctness (the DP body writes every cell in the
/// active region before reading) but a small perf hit (re-allocates on
/// next call).
pub fn reset_dp_pools() {
    DP_H_POOL.with_borrow_mut(|p| p.clear());
    DP_IJP_POOL.with_borrow_mut(|p| p.clear());
    DP_SCRATCH.with_borrow_mut(|s| *s = DpScratch::new());
}

/// Reset C-compat memoization on this thread. Engine calls this at the
/// start of each retree pass to mirror C's `Salignmm.c:1365-1366` where
/// `previousfirstlen = -1; previousicyc = -1;` is set after a buffer
/// resize (effectively per-pass).
pub fn reset_cpmx_memo() {
    CPMX_MEMO.with_borrow_mut(|s| {
        s.previous_call = false;
        s.previous_firstmem = -1;
        s.previous_icyc = 0;
        s.previous_first_len = 0;
    });
}

/// Add one new sequence's opening / closing gap counts to existing
/// per-thread accumulators, matching `st_OpeningGapAdd` /
/// `st_FinalGapAdd` in `mltaln9.c:12795,12906`. Scales existing by
/// `orieff`, adds `neweff`-weighted transitions for the new sequence.
fn cpmx_add_opening_closing(
    new_seq: &[u8],
    neweff: f64,
    orieff: f64,
    opening: &mut [f64],
    closing: &mut [f64],
    lgth: usize,
) {
    for j in 0..lgth {
        opening[j] *= orieff;
        closing[j] *= orieff;
    }
    let n = new_seq.len().min(lgth);
    let mut gc_prev = false;
    for pos in 0..n {
        let ch = unsafe { *new_seq.get_unchecked(pos) };
        let is_gap = ch == b'-' || ch == b'.';
        if is_gap {
            if !gc_prev { opening[pos] += neweff; }
        } else if gc_prev && pos > 0 {
            closing[pos - 1] += neweff;
        }
        gc_prev = is_gap;
    }
    if n < lgth && !gc_prev { opening[n] += neweff; }
    if n > 0 && gc_prev { closing[n - 1] += neweff; }
}

/// Grow `pool` so the first `rows` rows each hold at least `cols`
/// cells. No per-cell reset — every read cell in `h` / `ijp` is
/// unconditionally written by either the boundary init (`ijp[i][0]`,
/// `ijp[0][j]`, `h[i][0]`, `h[0][j]`) or by the DP body before any
/// traceback read, so stale data in unused cells (or in the
/// `[rows..][cols..]` tail beyond the active region) cannot affect
/// correctness. `fill` is the initial value for newly grown cells
/// (only relevant the FIRST time a row reaches a given length).
fn ensure_2d<T: Clone>(pool: &mut Vec<Vec<T>>, rows: usize, cols: usize, fill: T) {
    while pool.len() < rows {
        pool.push(Vec::new());
    }
    for row in pool.iter_mut().take(rows) {
        if row.len() < cols {
            row.resize(cols, fill.clone());
        }
    }
}

/// A position-specific frequency matrix (profile).
///
/// `freq[position][alphabet_index]` = weighted frequency of residue at that position.
#[derive(Debug, Clone)]
pub struct Profile {
    /// Frequency matrix: `freqs[pos][residue_idx]`.
    pub freqs: Vec<Vec<f64>>,
    /// Gap frequency at each position (0.0 = no gaps, 1.0 = all gaps).
    pub gap_freq: Vec<f64>,
    /// Non-gap frequency at each position (= 1.0 - gap_freq). Used for
    /// cross-weighting in gap cost computation.
    pub nongap_freq: Vec<f64>,
    /// Opening gap cost profile: weighted count of gap-opening transitions
    /// (non-gap → gap) at each position, modulated by penalty and nongap_freq.
    /// Formula: `ogcp[i] = 0.5 * (1.0 - opening_count[i]) * penalty * nongap_freq[i]`
    pub ogcp: Vec<f64>,
    /// Final gap cost profile: weighted count of gap-closing transitions
    /// (gap → non-gap) at each position.
    /// Formula: `fgcp[i] = 0.5 * (1.0 - closing_count[i]) * penalty * nongap_freq[i]`
    pub fgcp: Vec<f64>,
    /// Number of positions (alignment columns).
    pub length: usize,
    /// Alphabet size.
    pub nalphabets: usize,
}

impl Profile {
    /// Build a profile from a group of aligned sequences with weights.
    ///
    /// - `sequences`: aligned sequences (same length, with gaps as '-').
    /// - `weights`: per-sequence weights (must sum to ~1.0 for best results).
    /// - `amino_map`: ASCII char → internal residue index.
    /// - `nalphabets`: alphabet size (20 for protein, 26 with ambiguity).
    pub fn from_aligned(
        sequences: &[&[u8]],
        weights: &[f64],
        amino_map: &[u8; 256],
        nalphabets: usize,
    ) -> Self {
        assert_eq!(sequences.len(), weights.len());
        if sequences.is_empty() {
            return Self {
                freqs: Vec::new(),
                gap_freq: Vec::new(),
                nongap_freq: Vec::new(),
                ogcp: Vec::new(),
                fgcp: Vec::new(),
                length: 0,
                nalphabets,
            };
        }

        let length = sequences[0].len();
        let mut freqs = vec![vec![0.0f64; nalphabets]; length];
        let mut gap_freq = vec![0.0f64; length];

        // Opening gap count (non-gap → gap transitions) and final gap count
        // (gap → non-gap transitions), matching C's st_OpeningGapCount/st_FinalGapCount.
        let mut opening_count = vec![0.0f64; length];
        let mut closing_count = vec![0.0f64; length];

        // Single-pass fused loop: freqs + gap_freq + opening_count +
        // closing_count in one walk of each sequence (was three separate
        // walks before — measurable speedup in refinement-heavy modes like
        // FFT-NS-i where Profile::from_aligned is called O(nseq²) times).
        //
        // Semantics (preserved exactly from the original three-pass code):
        //
        //   - opening_count[pos] += w when the previous position was
        //     non-gap AND the current position IS gap (= a non-gap → gap
        //     transition at `pos`). The "previous position" before pos=0
        //     is treated as non-gap.
        //   - closing_count[pos] += w when the current position IS gap
        //     AND the next position is non-gap (= a gap → non-gap
        //     transition recorded at the gap's `pos`). The "next
        //     position" past the end is treated as non-gap.
        //   - For sequences shorter than `length`: positions [seq.len(),
        //     length) are treated as out-of-bounds. The original code
        //     read them as gaps in opening logic (`map_or(true, ...)`)
        //     and non-gaps in closing logic (`map_or(false, ...)`). We
        //     replicate that by treating an "out-of-bounds tail" as an
        //     immediate single gap (so opening fires once at seq.len() if
        //     the last in-bounds residue was non-gap; closing fires at
        //     seq.len() if it was a gap). Beyond seq.len() the state
        //     stays gap for the rest of the loop.
        for (seq, &w) in sequences.iter().zip(weights.iter()) {
            let n = seq.len().min(length);
            let mut gc_prev = false; // state at position -1: assume non-gap

            for pos in 0..n {
                let ch = unsafe { *seq.get_unchecked(pos) };
                let is_gap = ch == b'-' || ch == b'.';
                if is_gap {
                    gap_freq[pos] += w;
                    if !gc_prev {
                        opening_count[pos] += w;
                    }
                } else if gc_prev && pos > 0 {
                    // closing recorded at the GAP position (pos-1), not
                    // at the non-gap position we're currently visiting.
                    closing_count[pos - 1] += w;
                }
                let idx = amino_map[ch as usize] as usize;
                if idx < nalphabets {
                    freqs[pos][idx] += w;
                }
                gc_prev = is_gap;
            }

            // Tail beyond seq.len(): the original opening pass had
            // `seq.get(pos).map_or(true, ...)`, so the first out-of-bounds
            // position counts as a non-gap→gap transition iff the last
            // in-bounds residue was non-gap. (Beyond that, gc stays true
            // so no further opening_count[] firings.)
            if n < length && !gc_prev {
                opening_count[n] += w;
            }
            // Closing tail: original had `gc = seq.get(pos+1).map_or(false, ...)`
            // — past-the-end counts as non-gap, so if seq[n-1] is gap,
            // closing_count[n-1] fires. The branch above already records
            // closing at pos-1 when transitioning to non-gap; we still
            // need to flush a trailing gap run at the last in-bounds
            // position.
            if n > 0 && gc_prev {
                closing_count[n - 1] += w;
            }
        }

        // C: nongap_freq = 1.0 - gap_freq (no clamping).
        // C's convention is that weights sum to 1.0 (normalized before passing),
        // but we preserve C's exact formula even with unnormalized weights
        // to match cpmx_calc_new + gapcountf + st_*GapCount behavior exactly.
        let nongap_freq: Vec<f64> = gap_freq.iter().map(|&g| 1.0 - g).collect();

        // Store raw opening/closing counts; actual ogcp/fgcp are computed
        // in profile_align when the penalty parameter is known.
        Self {
            freqs,
            gap_freq,
            nongap_freq,
            ogcp: opening_count,
            fgcp: closing_count,
            length,
            nalphabets,
        }
    }

    /// `--c-compat` opt-in: build a profile using C's `cpmx_calc_add` /
    /// `st_OpeningGapAdd` / `st_FinalGapAdd` / `gapcountadd` semantics
    /// when the memo conditions match the prior call, OR fall back to
    /// `from_aligned` semantics otherwise.
    ///
    /// `firstmem`/`icyc`/`lgth` are C-style identifiers for the cluster
    /// being built: `firstmem` is the global index of the first member,
    /// `icyc` is the cluster size (number of sequences), `lgth` is the
    /// aligned-sequence length.
    ///
    /// Memo activates when: `previous_call == true && firstmem ==
    /// previous_firstmem && lgth == previous_first_len && icyc ==
    /// previous_icyc + 1`. Mirrors `Salignmm.c:1447`.
    ///
    /// When activated, only the **last** sequence (`sequences[icyc-1]`)
    /// is incrementally added to the per-thread persistent buffer via
    /// `cpmx_calc_add` semantics: existing freqs scaled by `orieff =
    /// 1.0 - neweff`, new sequence's residue gets `neweff` added per
    /// column. Same for opening / closing / gap_freq.
    pub fn from_aligned_with_memo(
        sequences: &[&[u8]],
        weights: &[f64],
        amino_map: &[u8; 256],
        nalphabets: usize,
        firstmem: i32,
        icyc: usize,
        lgth: usize,
    ) -> Self {
        assert_eq!(sequences.len(), weights.len());
        assert_eq!(sequences.len(), icyc);
        if sequences.is_empty() || lgth == 0 {
            return Self::from_aligned(sequences, weights, amino_map, nalphabets);
        }
        // C's `cpmx_calc_add` requires `lgth` to match the prior call's
        // and `icyc == previous_icyc + 1`. Otherwise fall back to
        // from-scratch — but we must also UPDATE the memo so the next
        // call sees correct previous_* values. (Hence we can't just
        // delegate to `from_aligned` for the else branch.)
        CPMX_MEMO.with_borrow_mut(|s| {
            let activate = s.previous_call
                && firstmem >= 0
                && firstmem == s.previous_firstmem
                && lgth == s.previous_first_len
                && icyc == s.previous_icyc + 1;
            let nalpha = nalphabets;
            // Ensure buffers sized to lgth × nalpha.
            if s.freqs.len() < nalpha {
                s.freqs.resize_with(nalpha, Vec::new);
            }
            for row in s.freqs.iter_mut().take(nalpha) {
                if row.len() < lgth {
                    row.resize(lgth, 0.0);
                }
            }
            if s.gap_freq.len() < lgth {
                s.gap_freq.resize(lgth, 0.0);
            }
            if s.opening.len() < lgth {
                s.opening.resize(lgth, 0.0);
            }
            if s.closing.len() < lgth {
                s.closing.resize(lgth, 0.0);
            }
            if activate {
                // `cpmx_calc_add` (tddis.c:160): scale existing by orieff,
                // add `neweff` to new sequence's slot per column.
                let newmem = icyc - 1;
                let neweff = weights[newmem];
                let orieff = 1.0 - neweff;
                let new_seq = sequences[newmem];
                for j in 0..lgth {
                    for i in 0..nalpha {
                        s.freqs[i][j] *= orieff;
                    }
                    if j < new_seq.len() {
                        let ch = new_seq[j];
                        let idx = amino_map[ch as usize] as usize;
                        if idx < nalpha {
                            s.freqs[idx][j] += neweff;
                        }
                    }
                }
                // `gapcountadd` (mltaln9.c:15109): mirror of gapcountf
                // for one new sequence: gap_freq[j] = orieff * gap_freq[j]
                // + neweff * (1 if gap at j else 0). Stored as gap freq
                // (NOT 1.0 - gap), C does the 1.0 - flip in A__align.
                for j in 0..lgth {
                    s.gap_freq[j] *= orieff;
                    if j < new_seq.len() && (new_seq[j] == b'-' || new_seq[j] == b'.') {
                        s.gap_freq[j] += neweff;
                    }
                }
                // `st_OpeningGapAdd` / `st_FinalGapAdd` (mltaln9.c:12795/12906):
                // incremental versions of *Count. Need same transition logic
                // as from_aligned but only for the newly added sequence,
                // scaled by orieff/neweff.
                cpmx_add_opening_closing(
                    new_seq, neweff, orieff, &mut s.opening, &mut s.closing, lgth);
            } else {
                // From-scratch: zero+fill (matches cpmx_calc_new etc.)
                for j in 0..lgth {
                    s.gap_freq[j] = 0.0;
                    s.opening[j] = 0.0;
                    s.closing[j] = 0.0;
                    for i in 0..nalpha { s.freqs[i][j] = 0.0; }
                }
                for (seq, &w) in sequences.iter().zip(weights.iter()) {
                    let n = seq.len().min(lgth);
                    let mut gc_prev = false;
                    for pos in 0..n {
                        let ch = unsafe { *seq.get_unchecked(pos) };
                        let is_gap = ch == b'-' || ch == b'.';
                        if is_gap {
                            s.gap_freq[pos] += w;
                            if !gc_prev { s.opening[pos] += w; }
                        } else if gc_prev && pos > 0 {
                            s.closing[pos - 1] += w;
                        }
                        let idx = amino_map[ch as usize] as usize;
                        if idx < nalpha { s.freqs[idx][pos] += w; }
                        gc_prev = is_gap;
                    }
                    if n < lgth && !gc_prev { s.opening[n] += w; }
                    if n > 0 && gc_prev { s.closing[n - 1] += w; }
                }
            }
            // Update memo state for next call.
            s.previous_call = true;
            s.previous_firstmem = firstmem;
            s.previous_icyc = icyc;
            s.previous_first_len = lgth;
            // Snapshot into a fresh Profile (lengths trimmed to current lgth).
            let mut freqs = vec![vec![0.0f64; nalpha]; lgth];
            for j in 0..lgth {
                for k in 0..nalpha { freqs[j][k] = s.freqs[k][j]; }
            }
            let gap_freq = s.gap_freq[..lgth].to_vec();
            let nongap_freq: Vec<f64> = gap_freq.iter().map(|&g| 1.0 - g).collect();
            Self {
                freqs,
                gap_freq,
                nongap_freq,
                ogcp: s.opening[..lgth].to_vec(),
                fgcp: s.closing[..lgth].to_vec(),
                length: lgth,
                nalphabets: nalpha,
            }
        })
    }

    /// Compute the match score between position `i` of this profile and
    /// position `j` of another profile, using the given scoring matrix.
    ///
    /// Uses a two-pass approach (matching C's `match_calc`) that enables
    /// SIMD auto-vectorization:
    /// 1. Build `scarr[b] = sum_a(freq1[a] * matrix[a][b])` — branchless
    /// 2. Dot-product `sum_b(scarr[b] * freq2[b])` — branchless, contiguous
    #[inline]
    pub fn match_score(
        &self,
        i: usize,
        other: &Profile,
        j: usize,
        matrix: &[Vec<f64>],
    ) -> f64 {
        let nalpha = self.nalphabets.min(other.nalphabets).min(matrix.len());
        let freq1 = &self.freqs[i];
        let freq2 = &other.freqs[j];

        // Use `mul_add` (single-rounding FMA) to match C's `gcc -O3` codegen
        // for `scarr[l] += matrix * cpmx`. Without FMA, accumulation rounds
        // twice per iteration and diverges from C by 1 ULP per term —
        // surfaces as anchor-selection differences for matrices with flat
        // score landscapes (e.g. `--tm 200 --bl 50`). See TODO §4/§5 close.
        let mut scarr = [0.0f64; 32]; // covers nalphabets <= 26
        for a in 0..nalpha {
            let f1 = freq1[a];
            let row = &matrix[a];
            let row_len = nalpha.min(row.len());
            for b in 0..row_len {
                scarr[b] = f1.mul_add(row[b], scarr[b]);
            }
        }

        let mut score = 0.0f64;
        for b in 0..nalpha {
            score = scarr[b].mul_add(freq2[b], score);
        }
        score
    }

    /// Extract a sub-profile (slice of positions from `start` to `end`).
    pub fn sub_profile(&self, start: usize, end: usize) -> Profile {
        let end = end.min(self.length);
        let start = start.min(end);
        Profile {
            freqs: self.freqs[start..end].to_vec(),
            gap_freq: self.gap_freq[start..end].to_vec(),
            nongap_freq: self.nongap_freq[start..end].to_vec(),
            ogcp: self.ogcp[start..end].to_vec(),
            fgcp: self.fgcp[start..end].to_vec(),
            length: end - start,
            nalphabets: self.nalphabets,
        }
    }
}

/// Align two profiles using anchor points, running DP within each segment.
///
/// Mirrors C's `Falign` segment-DP loop (`Falign.c:684-705`): builds cuts
/// `cut1 = [0, a1₀, a1₁, …, prof1.length]` and `cut2 = [0, a2₀, a2₁, …,
/// prof2.length]`, then runs `count - 1` independent profile DPs over
/// segments `[cut1[i], cut1[i+1]) × [cut2[i], cut2[i+1])`. Anchor
/// positions are segment BOUNDARIES — they are the START of the next
/// segment, NOT forced match cells. Within each segment the DP decides
/// match/gap freely.
///
/// Segment gap handling (Falign.c:686-687):
///   - First segment       (i == 0):           headgp = outgap, tailgp = 1
///   - Intermediate:                            headgp = 1,      tailgp = 1
///   - Last segment        (i == count - 2):   headgp = 1,      tailgp = outgap
///
/// With outgap=0 (the mafft script's `-O` for FFT-NS-2/L-INS-i/E-INS-i):
///   first → head_gap=false, last → tail_gap=false, middle → both true.
/// With outgap=1 (G-INS-i, --parttree — script omits `-O`):
///   ALL segments → head_gap=true, tail_gap=true.
///
/// Defaults to outgap=0; see `align_with_anchors_outgap` for the
/// outgap=1 variant used by `--parttree`.
pub fn align_with_anchors(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    anchors: &[(usize, usize)],
) -> Alignment {
    align_with_anchors_outgap(prof1, prof2, matrix, gap, anchors, false)
}

/// Variant of [`align_with_anchors`] that explicitly takes the C
/// `outgap` flag (`true` = penalize term gaps).
pub fn align_with_anchors_outgap(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    anchors: &[(usize, usize)],
    outgap: bool,
) -> Alignment {
    // Build cut1/cut2 mirroring C's Falign: bracket anchors with [0, …, length].
    let mut cut1: Vec<usize> = Vec::with_capacity(anchors.len() + 2);
    let mut cut2: Vec<usize> = Vec::with_capacity(anchors.len() + 2);
    cut1.push(0);
    cut2.push(0);
    for &(a1, a2) in anchors {
        // Skip anchors that lie at or past either profile end (boundary
        // collision with the trailing length cut). Also skip if the anchor
        // would not advance both cursors (defensive — shouldn't happen with
        // a well-formed anchor list from `find_fft_anchors` since
        // `block_align` enforces monotonic ordering).
        if a1 >= prof1.length || a2 >= prof2.length { continue; }
        let prev1 = *cut1.last().unwrap();
        let prev2 = *cut2.last().unwrap();
        if a1 < prev1 || a2 < prev2 { continue; }
        if a1 == prev1 && a2 == prev2 { continue; }
        cut1.push(a1);
        cut2.push(a2);
    }
    cut1.push(prof1.length);
    cut2.push(prof2.length);
    let count = cut1.len();

    let mut all_ops = Vec::new();
    let mut total_score = 0.0;

    for i in 0..count - 1 {
        let p1 = cut1[i];
        let p2 = cut2[i];
        let q1 = cut1[i + 1];
        let q2 = cut2[i + 1];

        // C's Falign.c:686-687:
        //   headgp = (i == 0) ? outgap : 1
        //   tailgp = (i == count-2) ? outgap : 1
        let head_gap = if i == 0 { outgap } else { true };
        let tail_gap = if i == count - 2 { outgap } else { true };

        if q1 > p1 || q2 > p2 {
            let sub1 = prof1.sub_profile(p1, q1);
            let sub2 = prof2.sub_profile(p2, q2);
            let seg_aln = profile_align(&sub1, &sub2, matrix, gap, head_gap, tail_gap);
            total_score += seg_aln.score;
            all_ops.extend(seg_aln.operations);
        }
    }

    Alignment {
        seq1: Vec::new(),
        seq2: Vec::new(),
        score: total_score,
        operations: all_ops,
    }
}

/// Align two profiles using affine gap DP.
///
/// Gap penalties are modulated by gap frequency: positions where many
/// sequences already have gaps receive reduced gap opening penalties.
///
/// Returns the alignment operations and score.
pub fn profile_align(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
) -> Alignment {
    profile_align_imp(prof1, prof2, matrix, gap, head_gap, tail_gap, None)
}

/// Profile alignment with optional per-cell importance bonuses.
///
/// `impmtx[i][j]` is added to the match score at DP cell `(i, j)`. Mirrors
/// C's `imp_match_out_vead` calls in `Salignmm.c::A__align` (lines 1700-1849)
/// where `currentw[j] += impmtx[i][j]` is applied before each row's cell
/// updates. Used by L-INS-i / E-INS-i to weight DP cells by local-homology
/// importance.
pub fn profile_align_imp(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
    impmtx: Option<&[Vec<f64>]>,
) -> Alignment {
    profile_align_imp_with_tiebreak(prof1, prof2, matrix, gap, head_gap, tail_gap, impmtx, false)
}

/// Boundary nongap-frequencies for partA__align-style segment alignment.
///
/// Mirrors C's `headgapfreq{1,2}` (sgap-derived; column [start-1] of full
/// alignment) and `gapfreq{1,2}[lgth]` (egap-derived; column [end] of full
/// alignment) used in `partSalignmm.c::partA__align` and `Salignmm.c::A__align`
/// when `sgap1/egap1` parameters are provided. All values are *nongap* fractions
/// in [0.0, 1.0] (after C's `1.0 - outgapcount` flip), defaulting to 1.0 when
/// the segment is at the global boundary.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryFreqs {
    pub head1: f64,
    pub head2: f64,
    pub tail1: f64,
    pub tail2: f64,
}

impl Default for BoundaryFreqs {
    fn default() -> Self {
        Self { head1: 1.0, head2: 1.0, tail1: 1.0, tail2: 1.0 }
    }
}

/// Clone a profile with `nongap_freq` overridden to all 1.0 and
/// `gap_freq` to all 0.0 — what C's `legacygapcost = 1` branch
/// (`Salignmm.c:1604-1610`) produces when computing the per-column
/// gap-frequency arrays for the profile DP.
fn legacy_gap_profile(p: &Profile) -> Profile {
    let mut clone = p.clone();
    clone.nongap_freq = vec![1.0; p.length];
    clone.gap_freq = vec![0.0; p.length];
    clone
}

/// Like `profile_align_imp` but with control over the prept-vs-mi/mjpt
/// tie-break rule. When `strict_part_tiebreak == false`, uses C's
/// `A__align` semantics (`>=`, ties update mi/mjpt — Salignmm.c:1926,1946).
/// When `true`, uses C's `partA__align` semantics (`>`, ties keep older
/// mi/mjpt — partSalignmm.c:1218,1235; commented "2018/Apr").
///
/// The refinement FFT-segmented constraint path (`Falign_localhom`) uses
/// `partA__align` per segment; the progressive constraint path uses
/// `A__align`. The DPs are otherwise identical.
pub fn profile_align_imp_with_tiebreak(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
    impmtx: Option<&[Vec<f64>]>,
    strict_part_tiebreak: bool,
) -> Alignment {
    profile_align_imp_with_boundary(
        prof1, prof2, matrix, gap, head_gap, tail_gap, impmtx,
        strict_part_tiebreak, BoundaryFreqs::default(),
    )
}

/// Profile DP with strict tie-break and explicit boundary nongap-frequencies.
///
/// Used by the refinement FFT-segmented constraint path so that interior
/// segments propagate the sgap/egap-derived nongap fractions of the column
/// just before/after the segment in the *full* alignment. Mirrors C's
/// `partA__align` (partSalignmm.c:678) when `sgap1/egap1/...` are non-NULL.
pub fn profile_align_imp_with_boundary(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
    impmtx: Option<&[Vec<f64>]>,
    strict_part_tiebreak: bool,
    boundary: BoundaryFreqs,
) -> Alignment {
    profile_align_imp_multimtx(prof1, prof2, matrix, gap, head_gap, tail_gap, impmtx, strict_part_tiebreak, boundary, None)
}

/// `profile_align_imp_with_boundary` with an optional multi-distance-class
/// match context (`--allowshift` refinement, C `partA__align_variousdist`).
/// When `multi` is `Some`, the per-cell substitution score comes from
/// [`crate::multimtx::MultiMtx::match_row`] (per-class cpmx add + spurious-pair
/// del) instead of the single `matrix`; everything else (gaps, imp, warp,
/// traceback) is identical. When `None`, behaviour is byte-for-byte the
/// original single-matrix path.
/// Batch `match_calc` for a single row of the profile DP — extracted
/// from `profile_align_imp_multimtx` so the body can be a plain function
/// (rather than a closure) and the caller can keep mutable ownership of
/// the pooled `scarr` buffer across calls. Mirrors C's `match_calc`
/// (Salignmm.c lines 216-223) plus the partA__align variant when
/// `multi` is `Some`. Writes into `output[0..m]`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn match_calc_row_into(
    row_pos: usize,
    output: &mut [f64],
    scarr: &mut [f64],
    multi: Option<&crate::multimtx::MultiMtx>,
    n: usize, m: usize, nalpha: usize,
    matrix: &[Vec<f64>],
    prof1_freqs: &[Vec<f64>],
    cpmx2_sparse: &[Vec<(usize, f64)>],
) {
    if let Some(mm) = multi {
        // Multi-distance-class match (C partA__align_variousdist). C's
        // calloc'd arrays give zeros for row_pos >= n (tail row).
        if row_pos >= n {
            for j in 0..m { output[j] = 0.0; }
        } else {
            // match_row_into reuses caller's buffer (avoids per-row
            // Vec allocation; hot path in --allowshift refinement).
            mm.match_row_into(row_pos, &mut output[..m]);
        }
        return;
    }
    if row_pos >= n {
        // Out-of-bounds: C's calloc'd arrays produce zeros here
        for j in 0..m {
            output[j] = 0.0;
        }
        return;
    }
    for l in 0..nalpha {
        scarr[l] = 0.0;
        for j in 0..nalpha {
            // C's match_calc compiled with `gcc -O3` fuses
            // `scarr[l] += a * b` into FMA (single-rounding fused
            // multiply-add). Rust's `+=` followed by `*` produces two
            // rounding steps. Use `mul_add` to match C's bit pattern.
            scarr[l] = matrix[j][l].mul_add(prof1_freqs[row_pos][j], scarr[l]);
        }
    }
    for j in 0..m {
        output[j] = 0.0;
        for &(k, v) in &cpmx2_sparse[j] {
            output[j] = scarr[k].mul_add(v, output[j]);
        }
    }
}

/// Const-generic specialisation of the profile DP inner j-loop body.
///
/// This is the hot kernel inside `profile_align_imp_multimtx` — called
/// once per outer-i row. The two `const bool` parameters lift what would
/// otherwise be per-cell runtime branches on `try_warp` and
/// `strict_part_tiebreak` into codegen-time choices:
///
/// - `TW = false` → the warp DP code is dead and LLVM strips it. This
///   removes ~25 lines of body, ~12 register pressures (warp state vars),
///   and the i-cache footprint of the warp arithmetic from the FFT-NS-i /
///   L/G/E-INS-i hot path. In that branch the warp slices arrive as
///   `&[]` / `&mut []` (cheap, valid, never accessed).
/// - `STRICT = true` → use strict `>` for the `mi`/`mj` tie-break
///   (mirrors C's `partA__align`). `false` → use `>=` (mirrors C's
///   `A__align`). Per-cell branch becomes a const choice.
///
/// The caller (the outer i-loop) dispatches once per row with
/// `match (try_warp, strict_part_tiebreak) { ... }` — a single
/// fully-predicted branch amortised over `m` cells.
///
/// SAFETY invariants (apply to every `get_unchecked` call below):
///   - `j` ranges `1..=m`.
///   - `prev_s`, `cur_s`, `ogcp2_s`, `fgcp2_s`, `mj_s`, `mpj_s`,
///     `h_row`, `ijp_row` are sized `m+1`, so indices in `{0..=m}` are
///     in-bounds.
///   - `prof2_ngf` is sized `m`; `j-1` always in `0..=m-1 < m`; the read
///     at index `j` is guarded `if j < m`.
///   - Under `TW = true`: `wmrecords`, `prevwmrecords`, `warpi`,
///     `warpj`, `prevwarpi`, `prevwarpj` are sized `m+1`; the
///     `warpis[warpn-1]` / `warpjs[warpn-1]` reads are inside the
///     `warpn > 0` guard.
///   - Under `TW = false`: every warp-state access is statically gated
///     behind `if TW { ... }` and never executes — the empty-slice args
///     are inert.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn j_loop<const TW: bool, const STRICT: bool>(
    m: usize, i: usize,
    h_row: &mut [f64], ijp_row: &mut [i32],
    cur_s: &mut [f64], prev_s: &[f64],
    ogcp2_s: &[f64], fgcp2_s: &[f64],
    mj_s: &mut [f64], mpj_s: &mut [usize],
    prof2_ngf: &[f64],
    gf1_i: f64, gf1_im1: f64,
    fgcp1_im1: f64, ogcp1_i: f64,
    boundary_tail2: f64, f_ext: f64,
    mi: &mut f64, mpi: &mut usize,
    // warp state (only touched when TW == true):
    fpenalty_shift: f64, warpbase: i32,
    wmrecords: &mut [f64], prevwmrecords: &[f64],
    warpi: &mut [i32], warpj: &mut [i32],
    prevwarpi: &[i32], prevwarpj: &[i32],
    warpis: &mut Vec<i32>, warpjs: &mut Vec<i32>,
    warpn: &mut usize,
) {
    let mut mi_v = *mi;
    let mut mpi_v = *mpi;
    for j in 1..=m {
        // Out-of-bounds positions in nongap_freq correspond to C's
        // gapfreq1[lgth1] / gapfreq2[lgth2] which take on egap-derived
        // values when partA__align is given sgap/egap
        // (boundary.tail{1,2}) and 1.0 otherwise.
        let gf2_j = if j < m {
            unsafe { *prof2_ngf.get_unchecked(j) }
        } else { boundary_tail2 };
        let gf2_jm1 = unsafe { *prof2_ngf.get_unchecked(j - 1) };

        let prevw_jm1 = unsafe { *prev_s.get_unchecked(j - 1) };
        let mut wm = prevw_jm1;
        unsafe { *ijp_row.get_unchecked_mut(j) = 0; }

        let g_jskip = unsafe { fgcp2_s.get_unchecked(j - 1).mul_add(gf1_i, mi_v) };
        if g_jskip > wm {
            wm = g_jskip;
            unsafe { *ijp_row.get_unchecked_mut(j) = -(j as i32 - mpi_v as i32); }
        }

        let g = unsafe { ogcp2_s.get_unchecked(j).mul_add(gf1_im1, prevw_jm1) };
        let mi_update = if STRICT { g > mi_v } else { g >= mi_v };
        if mi_update {
            mi_v = g;
            mpi_v = j - 1;
        }
        // C `Salignmm.c:1933`: `mi += fpenalty_ex;` — unconditional.
        mi_v += f_ext;

        let mj_j = unsafe { *mj_s.get_unchecked(j) };
        let g_iskip = fgcp1_im1.mul_add(gf2_j, mj_j);
        if g_iskip > wm {
            wm = g_iskip;
            unsafe {
                *ijp_row.get_unchecked_mut(j) =
                    i as i32 - *mpj_s.get_unchecked(j) as i32;
            }
        }

        let g = ogcp1_i.mul_add(gf2_jm1, prevw_jm1);
        let mj_update = if STRICT { g > mj_j } else { g >= mj_j };
        if mj_update {
            unsafe {
                *mj_s.get_unchecked_mut(j) = g;
                *mpj_s.get_unchecked_mut(j) = i - 1;
            }
        }
        // C `Salignmm.c:1953`: `m[j] += fpenalty_ex;` — unconditional.
        unsafe { *mj_s.get_unchecked_mut(j) += f_ext; }

        // Warp candidate (`Salignmm.c:1957-2003`). `if TW` is a
        // codegen-time choice; LLVM strips this entire block in the
        // TW=false specialisation.
        if TW {
            let pwi_jm1 = unsafe { *prevwarpi.get_unchecked(j - 1) };
            let pwj_jm1 = unsafe { *prevwarpj.get_unchecked(j - 1) };
            let pwm_jm1 = unsafe { *prevwmrecords.get_unchecked(j - 1) };
            let fpenalty_tmp = fpenalty_shift
                + f_ext * ((i as i32 - pwi_jm1) as f64
                         + (j as i32 - pwj_jm1) as f64);
            let g = pwm_jm1 + fpenalty_tmp;
            if g > wm {
                if *warpn > 0
                    && pwi_jm1 == unsafe { *warpis.get_unchecked(*warpn - 1) }
                    && pwj_jm1 == unsafe { *warpjs.get_unchecked(*warpn - 1) }
                {
                    unsafe { *ijp_row.get_unchecked_mut(j) = warpbase + (*warpn as i32) - 1; }
                } else {
                    unsafe { *ijp_row.get_unchecked_mut(j) = warpbase + (*warpn as i32); }
                    warpis.push(pwi_jm1);
                    warpjs.push(pwj_jm1);
                    *warpn += 1;
                }
                wm = g;
            }
        }

        unsafe {
            let c = cur_s.get_unchecked_mut(j);
            *c += wm;
            *h_row.get_unchecked_mut(j) = *c;
        }

        // Update wmrecords[j] / warpi[j] / warpj[j] (`Salignmm.c:1987-1998`).
        // TW=false strips this whole block too.
        if TW {
            unsafe {
                let wmr_jm1 = *wmrecords.get_unchecked(j - 1);
                let wmr_j = wmrecords.get_unchecked_mut(j);
                if wmr_jm1 > *wmr_j {
                    *wmr_j = wmr_jm1;
                    *warpi.get_unchecked_mut(j) = *warpi.get_unchecked(j - 1);
                    *warpj.get_unchecked_mut(j) = *warpj.get_unchecked(j - 1);
                }
                let curm = *cur_s.get_unchecked(j);
                if curm > *wmr_j {
                    *wmr_j = curm;
                    *warpi.get_unchecked_mut(j) = i as i32;
                    *warpj.get_unchecked_mut(j) = j as i32;
                }
            }
        }
    }
    *mi = mi_v;
    *mpi = mpi_v;
}

pub fn profile_align_imp_multimtx(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
    impmtx: Option<&[Vec<f64>]>,
    strict_part_tiebreak: bool,
    boundary: BoundaryFreqs,
    multi: Option<&crate::multimtx::MultiMtx>,
) -> Alignment {
    let n = prof1.length;
    let m = prof2.length;

    if n == 0 || m == 0 {
        return Alignment {
            seq1: Vec::new(),
            seq2: Vec::new(),
            score: 0.0,
            // An empty anchored segment must still consume the other profile.
            operations: if n == 0 { vec![AlignOp::Insert; m] } else { vec![AlignOp::Delete; n] },
        };
    }

    // `--leavegappyregion` / `--legacygappenalty` (`legacygapcost = 1`,
    // `Salignmm.c:1592-1610`): force every column to be treated as fully
    // nongap (`gapfreq[i] = 1.0`, head/tail nongap = 1.0). The cleanest
    // way to localise this is to clone the inputs with overridden
    // nongap-freq / gap-freq vectors and recurse with the legacy flag
    // cleared, so the rest of the DP body stays unchanged.
    if gap.legacy_gap_cost {
        let p1 = legacy_gap_profile(prof1);
        let p2 = legacy_gap_profile(prof2);
        let mut gap_clean = gap.clone();
        gap_clean.legacy_gap_cost = false;
        return profile_align_imp_with_boundary(
            &p1, &p2, matrix, &gap_clean,
            head_gap, tail_gap, impmtx, strict_part_tiebreak,
            BoundaryFreqs::default(),
        );
    }

    // Take all per-call scratch from the thread-local pool. The Vecs are
    // pre-allocated from prior calls, sized to whatever the largest input
    // on this thread has been so far. We then `clear()` + `push` or
    // `resize` each one as needed; the underlying buffer is reused when
    // the capacity already fits. Matches the C `A__align` static-TLS
    // amortisation. Put back into the pool at the bottom of the function.
    let DpScratch {
        mut ogcp1, mut fgcp1, mut ogcp2, mut fgcp2,
        mut initverticalw, mut currentw, mut previousw, mut lastverticalw,
        mut mj, mut mpj, mut scarr,
        mut cpmx1_sparse, mut cpmx2_sparse,
        mut wmrecords, mut prevwmrecords,
        mut warpi, mut warpj, mut prevwarpi, mut prevwarpj,
        mut warpis, mut warpjs,
    } = DP_SCRATCH.with_borrow_mut(std::mem::take);

    // Compute position-specific gap cost profiles matching C's formula:
    // ogcp[i] = 0.5 * (1.0 - opening_count[i]) * penalty * nongap_freq[i]
    // fgcp[i] = 0.5 * (1.0 - closing_count[i]) * penalty * nongap_freq[i]
    //
    // C allocates these arrays with size lgth+2 (via AllocateFloatVec),
    // so positions beyond the profile length are 0.0 (from calloc).
    // The DP loop accesses ogcp2[j] where j goes 1..lgth2, so the last
    // element accessed is ogcp2[lgth2] = 0.0. We add a trailing 0.0
    // to match this behavior. Pooled — `clear()` releases stale contents
    // but keeps the heap buffer; `push` reuses capacity when sufficient.
    let penalty = gap.open;
    ogcp1.clear(); ogcp1.reserve(n + 1);
    for i in 0..n { ogcp1.push(0.5 * (1.0 - prof1.ogcp[i]) * penalty * prof1.nongap_freq[i]); }
    ogcp1.push(0.0); // C's calloc padding
    fgcp1.clear(); fgcp1.reserve(n + 1);
    for i in 0..n { fgcp1.push(0.5 * (1.0 - prof1.fgcp[i]) * penalty * prof1.nongap_freq[i]); }
    fgcp1.push(0.0);
    ogcp2.clear(); ogcp2.reserve(m + 1);
    for j in 0..m { ogcp2.push(0.5 * (1.0 - prof2.ogcp[j]) * penalty * prof2.nongap_freq[j]); }
    ogcp2.push(0.0);
    fgcp2.clear(); fgcp2.reserve(m + 1);
    for j in 0..m { fgcp2.push(0.5 * (1.0 - prof2.fgcp[j]) * penalty * prof2.nongap_freq[j]); }
    fgcp2.push(0.0);

    // Exact port of C's MSalignmm_tanni (MSalignmm.c lines 770-888).
    // Uses C's exact rolling-row scheme, batch match_calc with sparse
    // dot product, and pointer-order-matching float accumulation.

    let hgf1: f64 = boundary.head1;
    let hgf2: f64 = boundary.head2;
    let gf1_0 = prof1.nongap_freq.first().copied().unwrap_or(1.0);
    let gf2_0 = prof2.nongap_freq.first().copied().unwrap_or(1.0);
    let nalpha = prof1.nalphabets.min(prof2.nalphabets).min(matrix.len());

    // Build sparse representation of prof2 (C's cpmxpd/cpmxpdn, lines 196-212).
    // For each position j, store only non-zero (alphabet_index, frequency) pairs.
    // Pooled: shrink/grow the outer Vec to length m; clear each inner Vec
    // (preserving its capacity) before refilling. For Vec<Vec<...>> this is
    // a big win — the inner allocations are the expensive ones.
    if cpmx2_sparse.len() < m {
        cpmx2_sparse.resize_with(m, Vec::new);
    } else {
        cpmx2_sparse.truncate(m);
    }
    for j in 0..m {
        let entries = &mut cpmx2_sparse[j];
        entries.clear();
        entries.reserve(nalpha);
        for l in 0..nalpha {
            let v = prof2.freqs[j][l];
            if v != 0.0 {
                entries.push((l, v));
            }
        }
    }

    // Ensure scarr has nalpha slots; cleared lazily per-row inside
    // `match_calc_row_into` (each cell is fully overwritten by the
    // `for l` loop before any read in the `for j` accumulator below).
    if scarr.len() < nalpha {
        scarr.resize(nalpha, 0.0);
    }

    // `match_calc_row_into` (module-level helper below) computes the
    // batch match-score row for row index `row_pos`, writing into
    // `output[0..m]`. Reusable scratch (`scarr`) is passed in by &mut
    // so the helper doesn't allocate.

    // h matrix (need full for traceback) and ijp.
    //
    // §C.2 path-(1): take these from thread-local pools (the dominant
    // allocations per call — ~3 MB for h, ~1.5 MB for ijp at typical
    // sizes). Grow-only resize, no per-cell zero (every read cell is
    // unconditionally written by the boundary init below or the DP
    // body before any traceback read). The pool is restored at the
    // end of the function before constructing the return value.
    let mut h: Vec<Vec<f64>> = DP_H_POOL.with_borrow_mut(std::mem::take);
    let mut ijp: Vec<Vec<i32>> = DP_IJP_POOL.with_borrow_mut(std::mem::take);
    ensure_2d(&mut h, n + 1, m + 1, 0.0f64);
    ensure_2d(&mut ijp, n + 1, m + 1, 0i32);

    // initverticalw (C line 776): match_calc with prof2 pos 0 vs all prof1 positions.
    // C calls match_calc with swapped profiles: cpmx2pt first, cpmx1pt second.
    // Pooled (same Vec<Vec<...>> reuse pattern as cpmx2_sparse above).
    if cpmx1_sparse.len() < n {
        cpmx1_sparse.resize_with(n, Vec::new);
    } else {
        cpmx1_sparse.truncate(n);
    }
    for i in 0..n {
        let entries = &mut cpmx1_sparse[i];
        entries.clear();
        entries.reserve(nalpha);
        for l in 0..nalpha {
            let v = prof1.freqs[i][l];
            if v != 0.0 {
                entries.push((l, v));
            }
        }
    }

    // initverticalw (C line 776): match_calc(cpmx2pt, cpmx1pt, 0, lgth1, initverticalw)
    // C fills initverticalw[0..lgth1-1] (0-based), then adds gap to [1..lgth1].
    // Result: [0] = pure match score (no gap), [1..n-1] = match + gap, [n] = 0 + gap.
    //
    // NOTE: uses `mul_add` for the same FMA-rounding reason as
    // `match_calc_row` above. Without this, --tm 200 / flat-landscape
    // matrices accumulate 1-ULP boundary differences that flip DP
    // tie-breaks in the first retree pass (§B.2).
    //
    // Pooled: resize-to-fill (n+1 zeros). Entry [n] is C's calloc padding;
    // entries [0..n] are explicitly overwritten by the if/else below.
    initverticalw.clear();
    initverticalw.resize(n + 1, 0.0);
    if let Some(mm) = multi {
        // C `partSalignmm.c:1764-1769`: fillzero + swapped per-class add/del.
        let col = mm.match_col(n);
        initverticalw[..n].copy_from_slice(&col[..n]);
    } else {
        // Reuse the pooled `scarr_buf` (already sized to nalpha and used
        // by `match_calc_row`). We're outside the closure's call window so
        // no aliasing concern. — Actually we can't; the closure captured
        // it by move. Use a small stack-allocated array via Vec instead.
        // The boundary init runs once per call so the alloc cost is
        // negligible vs the inner DP.
        let mut bcarr: Vec<f64> = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            bcarr[l] = 0.0;
            for j in 0..nalpha {
                bcarr[l] = matrix[j][l].mul_add(prof2.freqs[0][j], bcarr[l]);
            }
        }
        for i in 0..n {
            initverticalw[i] = 0.0;
            for &(k, v) in &cpmx1_sparse[i] {
                initverticalw[i] = bcarr[k].mul_add(v, initverticalw[i]);
            }
        }
    }
    // C: imp_match_out_vead_tate(initverticalw, 0, lgth1) — add impmtx[i][0] to initverticalw[i].
    if let Some(imp) = impmtx {
        for i in 0..n {
            initverticalw[i] += imp[i][0];
        }
    }
    if head_gap {
        for i in 1..=n {
            // C `Salignmm.c:1793`:
            //   initverticalw[i] += (ogcp1[0]*headgapfreq2 + fgcp1[i-1]*gapfreq2pt[0]);
            // Apple clang at -O3 with FP_CONTRACT=on lowers this to:
            //   fmul d3, fgcp1, gapfreq2pt0   ; t1 = fgcp1[i-1] * gf2_0 (plain mul)
            //   fmadd d0, ogcp1, hgf2, d3     ; t2 = ogcp1[0]*hgf2 + t1 (single FMA)
            //   fadd d0, initvert, d0         ; initverticalw += t2 (plain add)
            // The nested-mul_add form `fma(C,D, fma(A,B, init))` differs in
            // FP rounding from clang's `init + (A*B + C*D)` order — and that
            // 1-ULP boundary mismatch propagates into refinement DP
            // tie-breaks when `--exp` is non-zero (which adds a per-row
            // `fpenalty_ex * i` term that amplifies the drift). Match
            // clang's order exactly here, identical to the `currentw`
            // init below.
            let t1 = fgcp1[i - 1] * gf2_0;
            let t2 = ogcp1[0].mul_add(hgf2, t1);
            initverticalw[i] += t2;
            // C `Salignmm.c:1795`: `initverticalw[i] += fpenalty_ex * i;`
            initverticalw[i] = gap.extend.mul_add(i as f64, initverticalw[i]);
        }
    }

    // currentw (C line 779): match_calc with prof1 pos 0 vs all prof2 positions.
    // C fills currentw[0..lgth2-1] (0-based), then adds gap to [1..lgth2].
    // Result: [0] = pure match score (no gap), [1..m-1] = match + gap, [m] = 0 + gap.
    // Pooled.
    currentw.clear();
    currentw.resize(m + 1, 0.0);
    if let Some(mm) = multi {
        let row = mm.match_row(0, m);
        currentw[..m].copy_from_slice(&row[..m]);
    } else {
        // Same pattern as initverticalw above — small boundary-init buffer.
        let mut bcarr: Vec<f64> = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            bcarr[l] = 0.0;
            for j in 0..nalpha {
                bcarr[l] = matrix[j][l].mul_add(prof1.freqs[0][j], bcarr[l]);
            }
        }
        for j in 0..m {
            currentw[j] = 0.0;
            for &(k, v) in &cpmx2_sparse[j] {
                currentw[j] = bcarr[k].mul_add(v, currentw[j]);
            }
        }
    }
    // §E.1 forensic: dump currentw[0..5] BEFORE imp_match_out_vead so the
    // impmtx[0][·] contribution can be isolated (= post-imp minus pre-imp).
    // Mirrors C `Salignmm.c` `C_INITROW_PREIMP` instrumentation.
    if let Ok(path) = std::env::var("RS_H_DUMP") {
        let shape_ok = std::env::var("RS_H_DUMP_SHAPE")
            .ok()
            .and_then(|s| {
                let parts: Vec<&str> = s.split(',').collect();
                if parts.len() >= 2 {
                    let sn: usize = parts[0].parse().ok()?;
                    let sm: usize = parts[1].parse().ok()?;
                    Some(n == sn && m == sm)
                } else { None }
            })
            .unwrap_or(false);
        if shape_ok {
            use std::io::Write;
            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let lim = m.min(5);
                let _ = write!(fp, "R_INITROW_PREIMP");
                for k in 0..lim { let _ = write!(fp, " {:.17e}", currentw[k]); }
                let _ = writeln!(fp);
            }
        }
    }
    // C: imp_match_out_vead(currentw, 0, lgth2) — add impmtx[0][j] to currentw[j].
    if let Some(imp) = impmtx {
        for j in 0..m {
            currentw[j] += imp[0][j];
        }
    }
    if head_gap {
        for j in 1..=m {
            // C `Salignmm.c:1747`:
            //   currentw[j] += (ogcp2[0]*headgapfreq1 + fgcp2[j-1]*gapfreq1pt[0]);
            // Apple clang at -O3 with FP_CONTRACT=on lowers this to:
            //   fmul d3, fgcp2, gapfreq1pt0   ; t1 = fgcp2[j-1] * gapfreq1pt[0] (plain mul)
            //   fmadd d0, ogcp2, hgf1, d3     ; t2 = ogcp2[0]*hgf1 + t1 (single FMA)
            //   fadd d0, currentw, d0         ; currentw += t2 (plain add)
            let t1 = fgcp2[j - 1] * gf1_0;
            let t2 = ogcp2[0].mul_add(hgf1, t1);
            currentw[j] += t2;
            // C `Salignmm.c:1749`: `currentw[j] += fpenalty_ex * j;`
            currentw[j] = gap.extend.mul_add(j as f64, currentw[j]);
        }
    }

    for j in 0..=m { h[0][j] = currentw[j]; }
    for i in 0..=n { h[i][0] = initverticalw[i]; }

    // Pooled. The DP body reads mj[j] / mpj[j] only for j in 1..=m, so
    // mj[0]/mpj[0] never matter. mj[1..=m] is initialised by the loop
    // immediately below; mpj[1..=m] gets written there too.
    mj.clear();
    mj.resize(m + 1, f64::NEG_INFINITY);
    mpj.clear();
    mpj.resize(m + 1, 0);
    for j in 1..=m {
        let gf2_jm1 = prof2.nongap_freq.get(j - 1).copied().unwrap_or(0.0);
        // FMA: same rationale as the inner DP gap-candidate computations
        // above (see line 655). Without FMA the column tracker `mj[j]`
        // starts 1-ULP off C's value for flat-landscape matrices (TM
        // PAM 200), and that drift propagates into tie-break decisions.
        mj[j] = ogcp1[1].mul_add(gf2_jm1, currentw[j - 1]);
        mpj[j] = 0;
    }

    // §E.1-style FP-ordering forensic: dump the pre-loop DP state at
    // full %.17e precision so the C/Rust divergence source can be
    // pinned. Gated on RS_H_DUMP + RS_H_DUMP_SHAPE. Mirrors the C-side
    // dumps so each value can be diffed directly by prefix.
    if let Ok(path) = std::env::var("RS_H_DUMP") {
        use std::io::Write;
        let shape_ok = std::env::var("RS_H_DUMP_SHAPE")
            .ok()
            .and_then(|s| {
                let parts: Vec<&str> = s.split(',').collect();
                if parts.len() >= 2 {
                    let sn: usize = parts[0].parse().ok()?;
                    let sm: usize = parts[1].parse().ok()?;
                    Some(n == sn && m == sm)
                } else { None }
            })
            .unwrap_or(false);
        if shape_ok {
            if let Ok(mut fp) = std::fs::OpenOptions::new()
                .create(true).append(true).open(&path)
            {
                fn dump_vec(fp: &mut std::fs::File, label: &str, v: &[f64]) {
                    let _ = write!(fp, "{}", label);
                    for x in v { let _ = write!(fp, " {:.17e}", x); }
                    let _ = writeln!(fp);
                }
                fn dump_scalar(fp: &mut std::fs::File, label: &str, x: f64) {
                    let _ = writeln!(fp, "{} {:.17e}", label, x);
                }
                dump_vec(&mut fp, "R_INITROW", &currentw[..m]);
                dump_vec(&mut fp, "R_INITVERT", &initverticalw[..n]);
                dump_vec(&mut fp, "R_OGCP1", &ogcp1[..n]);
                dump_vec(&mut fp, "R_FGCP1", &fgcp1[..n]);
                dump_vec(&mut fp, "R_OGCP2", &ogcp2[..m]);
                dump_vec(&mut fp, "R_FGCP2", &fgcp2[..m]);
                dump_vec(&mut fp, "R_NONGAP1", &prof1.nongap_freq[..n]);
                dump_vec(&mut fp, "R_NONGAP2", &prof2.nongap_freq[..m]);
                dump_scalar(&mut fp, "R_FPENALTY", penalty);
                dump_scalar(&mut fp, "R_FPENALTY_EX", gap.extend);
                dump_scalar(&mut fp, "R_FPENALTY_SHIFT", gap.shift.unwrap_or(0.0));
                dump_scalar(&mut fp, "R_HGF1", hgf1);
                dump_scalar(&mut fp, "R_HGF2", hgf2);
                dump_vec(&mut fp, "R_MJ_INIT", &mj[1..=m]);
                // impmtx is added to currentw inside the if-let above (line
                // ~963). Dump impmtx[0][0..5] and impmtx[i][0] for a few i
                // to compare against C's imp_match_out_vead behaviour.
                if let Some(imp) = impmtx {
                    if !imp.is_empty() {
                        let row0 = &imp[0];
                        let n0 = row0.len().min(5);
                        dump_vec(&mut fp, "R_IMP_ROW0_HEAD", &row0[..n0]);
                    }
                    let nrows = imp.len().min(5);
                    let mut col0: Vec<f64> = Vec::with_capacity(nrows);
                    for i in 0..nrows {
                        col0.push(imp[i].first().copied().unwrap_or(0.0));
                    }
                    dump_vec(&mut fp, "R_IMP_COL0_HEAD", &col0);
                }
                // Verify the cpmx col-0 weight at residue 12 (M) is exactly 1.0
                // for our 1-seq cluster; if not, weight scaling is the source.
                if !prof1.freqs.is_empty() && prof1.freqs[0].len() > 12 {
                    dump_scalar(&mut fp, "R_PROF1_F0_M", prof1.freqs[0][12]);
                }
                if !prof2.freqs.is_empty() && prof2.freqs[0].len() > 12 {
                    dump_scalar(&mut fp, "R_PROF2_F0_M", prof2.freqs[0][12]);
                }
            }
        }
    }

    for i in 0..=n { ijp[i][0] = i as i32 + 1; }
    for j in 0..=m { ijp[0][j] = -(j as i32 + 1); }

    // Pooled. previousw is fully overwritten by the i-loop swap (mem::swap
    // with currentw), then index 0 is set from initverticalw[i-1]; the rest
    // is overwritten by match_calc_row. The first iteration's swap makes
    // previousw = (the freshly-built currentw); the initial contents of
    // previousw[1..m] are never read. Zero-fill is cosmetic but cheap.
    previousw.clear();
    previousw.resize(m + 1, 0.0);
    // lastverticalw[0] is set immediately below; lastverticalw[i] is set
    // each row (`lastverticalw[i] = currentw[m - 1];`). Resize-with-zero
    // is fine.
    lastverticalw.clear();
    lastverticalw.resize(n + 1, 0.0);
    lastverticalw[0] = currentw[m - 1];

    // Warp DP state (`Salignmm.c:1255-1281,1888-2022`). Activates when
    // `gap.shift` is `Some` (penalty_shift_factor < 10 → trywarp = 1). Mirrors
    // the same recurrence already ported into `global.rs` for `G__align11`.
    // Required for `--allowshift` byte-identity in the progressive merge.
    let try_warp = gap.shift.is_some();
    let fpenalty_shift = gap.shift.unwrap_or(0.0);
    let f_ext = gap.extend;
    let warpbase: i32 = (n + m) as i32;
    let neg_warpbase: i32 = -warpbase;
    let mut warpn: usize = 0;
    // Pooled. `warpis` / `warpjs` start empty and grow as warp anchors are
    // discovered; previous-call entries must NOT carry over. `clear()`
    // truncates length to 0 while keeping the heap buffer (capacity), so
    // subsequent `push` calls don't reallocate as long as the anchor count
    // doesn't exceed the high-water mark.
    warpis.clear();
    warpjs.clear();
    // C uses `AllocateFloatVec` (calloc) → zero-init, then explicitly sets
    // `wmrecords[i] = 0.0` / `prevwmrecords[i] = 0.0` (Salignmm.c:1276-1277).
    //
    // Pooled. When `try_warp == false` the inner loop's warp paths are
    // statically gated (TW=false specialisation strips them), so the
    // get_unchecked reads against empty Vecs are never executed. We leave
    // the pooled buffers untouched in that case — no work at all.
    // When `try_warp == true`, resize/fill each buffer (initialising to
    // 0.0 / neg_warpbase as C does).
    if try_warp {
        wmrecords.clear();
        wmrecords.resize(m + 1, 0.0);
        prevwmrecords.clear();
        prevwmrecords.resize(m + 1, 0.0);
        warpi.clear();
        warpi.resize(m + 1, neg_warpbase);
        warpj.clear();
        warpj.resize(m + 1, neg_warpbase);
        prevwarpi.clear();
        prevwarpi.resize(m + 1, neg_warpbase);
        prevwarpj.clear();
        prevwarpj.resize(m + 1, neg_warpbase);
    } else {
        // Force-empty so the inner-loop slices (`&mut wmrecords` etc.) are
        // zero-length. The TW=false specialisation never reads them, but
        // zero-length is the only safe state for a never-read slice.
        wmrecords.clear();
        prevwmrecords.clear();
        warpi.clear();
        warpj.clear();
        prevwarpi.clear();
        prevwarpj.clear();
    }

    // C pads gapfreq1pt[lgth1] = 1.0 and gapfreq2pt[lgth2] = 1.0 (tditeration.c's
    // `for(i=0;i<lgth+1;i++) gapfreq[i] = 1.0 - gapfreq[i];` with calloc'd 0 → 1).
    // Our DP needs these boundary values when i==n or j==m.
    let lasti = if tail_gap { n + 1 } else { n };

    for i in 1..lasti {
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[0] = initverticalw[i - 1];

        // Batch match_calc for row i (C line 816: ist+i)
        // Fills currentw[0..m-1]. Position m doesn't exist in prof2.
        match_calc_row_into(
            i, &mut currentw, &mut scarr[..nalpha],
            multi, n, m, nalpha, matrix, &prof1.freqs, &cpmx2_sparse,
        );
        // C: imp_match_out_vead(currentw, i, lgth2) — add impmtx[i][:] (Salignmm.c:1848).
        if let Some(imp) = impmtx {
            if i < imp.len() {
                let row = &imp[i];
                for j in 0..m.min(row.len()) {
                    currentw[j] += row[j];
                }
            }
        }
        currentw[m] = 0.0; // padding: no substitution score beyond prof2
        currentw[0] = initverticalw[i];

        let gf1_im1 = prof1.nongap_freq[i - 1]; // i-1 in 0..n-1, always valid
        // Per-row invariants — gf1_i, gf1_im1, fgcp1_im1, ogcp1_i do not
        // depend on j; hoist out of the inner loop to save bounds checks +
        // branch per cell. For long profiles (~hundreds of residues) this
        // is a measurable win. (fgcp1_im1 / ogcp1_i were previously read
        // inside the j-loop and re-loaded every iteration.)
        let gf1_i = if i < n { prof1.nongap_freq[i] } else { boundary.tail1 };
        let fgcp1_im1 = fgcp1[i - 1];
        let ogcp1_i = ogcp1[i];
        // Bind h[i] / ijp[i] to local mutable slices: hoists the outer-Vec
        // bounds check out of the inner loop. Same for h[i-1] (read by the
        // dump path only; not needed inside the hot loop since currentw
        // already carries the row-i score). Per-cell inner-Vec bounds
        // checks are still present but the outer-Vec ones are amortized
        // to once per row instead of M times per row.
        // SAFETY: i < lasti <= n+1 and h/ijp are sized (n+1) x (m+1).
        // CONFIRMED MICRO-WIN ~ a few % on long FFT-NS-i inputs.
        let h_row: &mut [f64] = &mut h[i];
        let ijp_row: &mut [i32] = &mut ijp[i];
        // Bind shared cold slices to locals so the inner loop sees them
        // as `&[T]` of statically-known length-bound rather than re-deriving
        // a slice pointer per access. This is purely about giving LLVM
        // enough info to eliminate the per-access bounds check in concert
        // with the `get_unchecked` calls below.
        let prof2_ngf: &[f64] = &prof2.nongap_freq;
        let ogcp2_s: &[f64] = &ogcp2;
        let fgcp2_s: &[f64] = &fgcp2;
        let mj_s: &mut [f64] = &mut mj;
        let mpj_s: &mut [usize] = &mut mpj;
        let prev_s: &mut [f64] = &mut previousw;
        let cur_s: &mut [f64] = &mut currentw;
        // SAFETY (the entire inner j-loop, inside j_loop):
        //   - see j_loop's SAFETY doc-block.
        // Use mul_add throughout — C compiled with `gcc -O3 -mfma` (or
        // equivalent) fuses `a + b * c` into FMA (single-rounding step);
        // matching this behavior is required for bit-identity to C's
        // A__align inner DP, which surfaces as tie-break divergences for
        // matrices with flatter score landscapes (e.g. BL50).
        let mut mi = unsafe {
            ogcp2_s.get_unchecked(1).mul_add(gf1_im1, *prev_s.get_unchecked(0))
        };
        let mut mpi: usize = 0;

        // Dispatch the j-loop body to a const-generic specialisation.
        // The 2×2 matrix of (TW, STRICT) lets LLVM strip the dead warp
        // path entirely in the FFT-NS-i / L/G/E-INS-i case and replace
        // the per-cell `if strict_part_tiebreak` branch with a constant
        // choice. Dispatch cost is one fully-predicted branch per row,
        // amortised over `m` inner cells.
        match (try_warp, strict_part_tiebreak) {
            (false, false) => j_loop::<false, false>(
                m, i, h_row, ijp_row, cur_s, prev_s,
                ogcp2_s, fgcp2_s, mj_s, mpj_s, prof2_ngf,
                gf1_i, gf1_im1, fgcp1_im1, ogcp1_i,
                boundary.tail2, f_ext,
                &mut mi, &mut mpi,
                fpenalty_shift, warpbase,
                &mut wmrecords, &prevwmrecords,
                &mut warpi, &mut warpj,
                &prevwarpi, &prevwarpj,
                &mut warpis, &mut warpjs, &mut warpn,
            ),
            (false, true) => j_loop::<false, true>(
                m, i, h_row, ijp_row, cur_s, prev_s,
                ogcp2_s, fgcp2_s, mj_s, mpj_s, prof2_ngf,
                gf1_i, gf1_im1, fgcp1_im1, ogcp1_i,
                boundary.tail2, f_ext,
                &mut mi, &mut mpi,
                fpenalty_shift, warpbase,
                &mut wmrecords, &prevwmrecords,
                &mut warpi, &mut warpj,
                &prevwarpi, &prevwarpj,
                &mut warpis, &mut warpjs, &mut warpn,
            ),
            (true, false) => j_loop::<true, false>(
                m, i, h_row, ijp_row, cur_s, prev_s,
                ogcp2_s, fgcp2_s, mj_s, mpj_s, prof2_ngf,
                gf1_i, gf1_im1, fgcp1_im1, ogcp1_i,
                boundary.tail2, f_ext,
                &mut mi, &mut mpi,
                fpenalty_shift, warpbase,
                &mut wmrecords, &prevwmrecords,
                &mut warpi, &mut warpj,
                &prevwarpi, &prevwarpj,
                &mut warpis, &mut warpjs, &mut warpn,
            ),
            (true, true) => j_loop::<true, true>(
                m, i, h_row, ijp_row, cur_s, prev_s,
                ogcp2_s, fgcp2_s, mj_s, mpj_s, prof2_ngf,
                gf1_i, gf1_im1, fgcp1_im1, ogcp1_i,
                boundary.tail2, f_ext,
                &mut mi, &mut mpi,
                fpenalty_shift, warpbase,
                &mut wmrecords, &prevwmrecords,
                &mut warpi, &mut warpj,
                &prevwarpi, &prevwarpj,
                &mut warpis, &mut warpjs, &mut warpn,
            ),
        }
        lastverticalw[i] = currentw[m - 1];

        // End of row: snapshot wmrecords/warpi/warpj to prev*
        // (`Salignmm.c:2017-2022`, `fltncpy(prevwmrecords, wmrecords, lastj)`
        // where lastj = lgth2 + 1). copy_from_slice lowers to memcpy.
        if try_warp {
            prevwmrecords.copy_from_slice(&wmrecords);
            prevwarpi.copy_from_slice(&warpi);
            prevwarpj.copy_from_slice(&warpj);
        }
    }

    if let Ok(path) = std::env::var("RS_H_DUMP") {
        use std::io::Write;
        let shape_ok = std::env::var("RS_H_DUMP_SHAPE")
            .ok()
            .and_then(|s| {
                let parts: Vec<&str> = s.split(',').collect();
                if parts.len() == 4 {
                    let sn: usize = parts[0].parse().ok()?;
                    let sm: usize = parts[1].parse().ok()?;
                    Some(n == sn && m == sm)
                } else { None }
            })
            .unwrap_or(false);
        if shape_ok {
            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                // Dump prof1.freqs[0] and prof2.freqs[0] for input diffing.
                let nalpha = prof1.nalphabets.min(prof2.nalphabets).min(matrix.len());
                let _ = write!(fp, "R_PROF1_F0");
                for k in 0..nalpha { let _ = write!(fp, " {:.17e}", prof1.freqs[0][k]); }
                let _ = writeln!(fp);
                let _ = write!(fp, "R_PROF2_F0");
                for k in 0..nalpha { let _ = write!(fp, " {:.17e}", prof2.freqs[0][k]); }
                let _ = writeln!(fp);
                let _ = write!(fp, "R_MATRIX_R0");
                for k in 0..nalpha { let _ = write!(fp, " {:.17e}", matrix[0][k]); }
                let _ = writeln!(fp);
                // Row 12 = M (col-0 residue of both BB12003 seq3/seq4):
                // if MATRIX_R0 matches but R12 differs, makedynamicmtx is
                // applying a non-uniform shift; if R12 also matches, the
                // divergence is in the match_calc summation order.
                let _ = write!(fp, "R_MATRIX_R12");
                for k in 0..nalpha { let _ = write!(fp, " {:.17e}", matrix[12][k]); }
                let _ = writeln!(fp);
                // EFF cannot be dumped here — profile_align_imp_with_boundary
                // doesn't see the weights vector. They're captured upstream
                // (refinement.rs) into prof.freqs.
                for ii in 1..=n {
                    let _ = write!(fp, "R_H i={}", ii);
                    for jj in 0..m {
                        let _ = write!(fp, " {:.17e}", h[ii][jj]);
                    }
                    let _ = writeln!(fp);
                }
            }
        }
    }

    // Tail gap handling.
    //
    // C has TWO different scan implementations:
    //   - `Atracking` (Salignmm.c:891-925, no-constraint path): strict `>`,
    //     wm initialized BELOW corner, scans row+col EXCLUDING corner with
    //     decreasing indices, then explicit corner fallback. Among tied
    //     cells the HIGHEST index wins; corner wins only if strictly greater.
    //   - `Atracking_localhom` (Salignmm.c:434-457, constraint path): `>=`,
    //     wm initialized to lastverticalw[0], scans col+row INCLUDING corner
    //     with increasing indices. Among tied cells the LATEST update wins
    //     — i.e. the corner (scanned last in the row pass) wins all ties.
    //
    // The progressive merge for L-INS-i / G-INS-i / E-INS-i uses the
    // constraint path (`A__align` with constraint != 0), so we must mirror
    // `Atracking_localhom` when `impmtx` is supplied. The non-constraint
    // path keeps `Atracking` semantics — this was the BL50 step 24 fix.
    //
    // TERMGAPFAC and TERMGAPFAC_EX are both 0.0 (Salignmm.c:12-13), so the
    // additive correction terms drop out.
    if !tail_gap {
        let last_row_corner = h[n - 1][m - 1];
        if impmtx.is_some() {
            // Atracking_localhom port: forward scan, `>=`, includes corner.
            let mut wm = lastverticalw[0];
            for i in 0..n {
                if lastverticalw[i] >= wm {
                    wm = lastverticalw[i];
                    ijp[n][m] = (n - i) as i32;
                }
            }
            for j in 0..m {
                if h[n - 1][j] >= wm {
                    wm = h[n - 1][j];
                    ijp[n][m] = -((m - j) as i32);
                }
            }
            h[n][m] = wm;
        } else {
            // Atracking port: reverse scan, strict `>`, excludes corner,
            // then corner fallback.
            let mut wm = last_row_corner - 1.0;
            for j in (0..m - 1).rev() {
                let g = h[n - 1][j];
                if g > wm {
                    wm = g;
                    ijp[n][m] = -((m - j) as i32);
                }
            }
            for i in (0..n - 1).rev() {
                let g = lastverticalw[i];
                if g > wm {
                    wm = g;
                    ijp[n][m] = (n - i) as i32;
                }
            }
            if last_row_corner > wm {
                wm = last_row_corner;
                ijp[n][m] = 0;
            }
            h[n][m] = wm;
        }
    }

    // Traceback  (C lines 581-617)
    let mut gaptable1 = Vec::new(); // 'o' = content, '-' = gap
    let mut gaptable2 = Vec::new();

    let (mut iin, mut jin) = (n as i32, m as i32);
    let klim = n + m;
    let mut k = 0;
    while k <= klim {
        let (ifi, jfi): (i32, i32);
        let v = ijp[iin as usize][jin as usize];
        if v >= warpbase {
            // Warp transition (`Salignmm.c:479-482`). Jump to anchor
            // (warpis[idx], warpjs[idx]).
            let idx = (v - warpbase) as usize;
            ifi = warpis[idx];
            jfi = warpjs[idx];
        } else if v < 0 {
            // Deletion: skip columns
            ifi = iin - 1;
            jfi = jin + v;
        } else if v > 0 {
            // Insertion: skip rows
            ifi = iin - v;
            jfi = jin - 1;
        } else {
            // Diagonal
            ifi = iin - 1;
            jfi = jin - 1;
        }

        // C `Salignmm.c:496-514`: handle uninitialized warp source by
        // emitting all remaining iin/jin residues as gaps and exiting.
        if v >= warpbase && ifi == neg_warpbase && jfi == neg_warpbase {
            let mut l = iin;
            while l > 0 {
                l -= 1;
                gaptable1.push(b'o');
                gaptable2.push(b'-');
            }
            let mut l = jin;
            while l > 0 {
                l -= 1;
                gaptable1.push(b'-');
                gaptable2.push(b'o');
            }
            break;
        }

        // Emit gap in prof2 for skipped rows (insertion in prof1).
        // For warp: emit (iin - ifi - 1) cells (no extra for first cell).
        // For non-warp gap-k: emit (k-1) cells, with `l > 1` condition.
        let mut l = iin - ifi;
        while l > 1 {
            gaptable1.push(b'o');
            gaptable2.push(b'-');
            k += 1;
            l -= 1;
        }
        // Emit gap in prof1 for skipped columns (deletion in prof2)
        let mut l = jin - jfi;
        while l > 1 {
            gaptable1.push(b'-');
            gaptable2.push(b'o');
            k += 1;
            l -= 1;
        }

        if iin <= 0 || jin <= 0 { break; }
        // Emit diagonal match
        gaptable1.push(b'o');
        gaptable2.push(b'o');
        k += 1;
        iin = ifi;
        jin = jfi;
    }

    // Convert gaptable (built in reverse) to AlignOps
    gaptable1.reverse();
    gaptable2.reverse();

    let mut ops = Vec::with_capacity(gaptable1.len());
    for k in 0..gaptable1.len() {
        match (gaptable1[k], gaptable2[k]) {
            (b'o', b'o') => ops.push(AlignOp::Match),
            (b'o', b'-') => ops.push(AlignOp::Delete),  // gap in prof2
            (b'-', b'o') => ops.push(AlignOp::Insert),  // gap in prof1
            _ => {}
        }
    }

    let best_score = h[n][m];

    if let Ok(f) = std::env::var("RS_DP_CORNER") {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_NO: AtomicUsize = AtomicUsize::new(0);
        let cn = CALL_NO.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
            let corner = h[n - 1][m - 1];
            let _ = writeln!(fp, "call={} constraint={} wm={:.18e} h[n-1][m-1]={:.18e} h[n][m]={:.18e}",
                cn, if impmtx.is_some() {1} else {0}, best_score, corner, h[n][m]);
        }
    }
    if let Ok(prefix) = std::env::var("RS_IJP_DUMP_PREFIX") {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_NO2: AtomicUsize = AtomicUsize::new(0);
        let cn = CALL_NO2.fetch_add(1, Ordering::SeqCst);
        let target_call: usize = std::env::var("RS_IJP_DUMP_CALL")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        if cn == target_call {
            let fname = format!("{}_call_{}.txt", prefix, cn);
            if let Ok(mut fp) = std::fs::File::create(&fname) {
                let _ = writeln!(fp, "n={} m={} corner_h={:.18e} best={:.18e}", n, m, h[n-1][m-1], best_score);
                for i in 0..=n {
                    for j in 0..=m {
                        let _ = writeln!(fp, "ijp[{}][{}]={} h[{}][{}]={:.18e}", i, j, ijp[i][j], i, j, h[i][j]);
                    }
                }
            }
        }
    }

    // Return h/ijp to the thread-local pools so the next call avoids
    // re-allocating them. (See the §C.2 path-(1) comment at the top of
    // the function for rationale.) Swap-in/swap-out matches C
    // `A__align`'s `static TLS` buffer amortisation strategy without
    // breaking determinism — h/ijp are write-then-read inside this
    // function, so stale data carried across calls cannot leak into
    // the alignment.
    DP_H_POOL.with_borrow_mut(|p| std::mem::swap(p, &mut h));
    DP_IJP_POOL.with_borrow_mut(|p| std::mem::swap(p, &mut ijp));

    // Return the scratch Vecs to the thread-local pool too. Each one
    // keeps its current capacity, so the next call on this thread
    // benefits from amortised allocation when input sizes are
    // comparable. The `match_calc_row` closure consumed `scarr` and
    // `cpmx2_sparse` by move (it had to so the closure could carry the
    // mutable scarr borrow); recover them by dropping the closure and
    // putting the moved values back via the helpers' returns. Actually
    // simpler: we already destructured the pool at function entry, so
    // here we only need to put each named local back.
    DP_SCRATCH.with_borrow_mut(|s| {
        *s = DpScratch {
            ogcp1, fgcp1, ogcp2, fgcp2,
            initverticalw, currentw, previousw, lastverticalw,
            mj, mpj, scarr,
            cpmx1_sparse, cpmx2_sparse,
            wmrecords, prevwmrecords,
            warpi, warpj, prevwarpi, prevwarpj,
            warpis, warpjs,
        };
    });

    Alignment {
        seq1: Vec::new(),
        seq2: Vec::new(),
        score: best_score,
        operations: ops,
    }
}

/// Pairwise alignment for single-sequence pairs.
///
/// Ports C's `G__align11()` from Galign11.c. Uses character-indexed scoring
/// matrix and flat gap penalties (no position-specific ogcp/fgcp weighting).
/// C uses this instead of MSalignmm when both groups have exactly 1 sequence.
pub fn pairwise_align11(
    seq1: &[u8],
    seq2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
    head_gap: bool,
    tail_gap: bool,
) -> Alignment {
    pairwise_align11_ex(
        seq1, seq2, matrix, amino_map, penalty, 0.0, head_gap, tail_gap,
    )
}

/// Extended form of [`pairwise_align11`] that takes a separate
/// gap-extension penalty (`penalty_ex`, C `fpenalty_ex`). Mirrors
/// C `Galign11.c::G__align11` lines 1362-1384 which add
/// `fpenalty_ex_i` to `mi` and `fpenalty_ex` to `m[j]` after the
/// diagonal update each cell. With `penalty_ex = 0` this is
/// identical to `pairwise_align11`.
///
/// Closes R-1b: without this per-cell gap-extension contribution,
/// `--nofft --exp > 0` diverged from C at tied-trace gap positions
/// (the missing `penalty_ex` accumulation changed which cells were
/// chosen as gap-region maxima).
pub fn pairwise_align11_ex(
    seq1: &[u8],
    seq2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    penalty: f64,
    penalty_ex: f64,
    head_gap: bool,
    tail_gap: bool,
) -> Alignment {
    let n = seq1.len();
    let m = seq2.len();
    if n == 0 || m == 0 {
        return Alignment { seq1: Vec::new(), seq2: Vec::new(), score: 0.0, operations: Vec::new() };
    }

    let nalpha = matrix.len();

    // match_calc_mtx: score = amino_dynamicmtx[seq1[i]][seq2[j]]
    // Which equals matrix[amino_map[seq1[i]]][amino_map[seq2[j]]]
    let score_pair = |c1: u8, c2: u8| -> f64 {
        let i = amino_map[c1 as usize] as usize;
        let j = amino_map[c2 as usize] as usize;
        if i < nalpha && j < nalpha { matrix[i][j] } else { 0.0 }
    };

    // initverticalw: match_calc_mtx(seq2, seq1, 0, lgth1)
    // C fills initverticalw[0..lgth1-1] (0-based), then adds gap to [1..lgth1].
    // Result: [0] = pure match score (no gap), [1..n-1] = match + gap, [n] = 0 + gap.
    let mut initverticalw = vec![0.0f64; n + 1];
    for i in 0..n {
        initverticalw[i] = score_pair(seq1[i], seq2[0]);
    }
    if head_gap {
        for i in 1..=n {
            initverticalw[i] += penalty; // flat penalty, C line 1180
        }
    }

    // currentw: match_calc_mtx(seq1, seq2, 0, lgth2)
    // C fills currentw[0..lgth2-1] (0-based), then adds gap to [1..lgth2].
    let mut currentw = vec![0.0f64; m + 1];
    for j in 0..m {
        currentw[j] = score_pair(seq1[0], seq2[j]);
    }
    if head_gap {
        for j in 1..=m {
            currentw[j] += penalty; // flat penalty, C line 1188
        }
    }

    let mut h = vec![vec![0.0f64; m + 1]; n + 1];
    let mut ijp = vec![vec![0i32; m + 1]; n + 1];

    for j in 0..=m { h[0][j] = currentw[j]; }
    for i in 0..=n { h[i][0] = initverticalw[i]; }

    // m[j] = currentw[j-1], NO ogcp multiplication (C line 1218)
    let mut mj = vec![f64::NEG_INFINITY; m + 1];
    let mut mpj = vec![0usize; m + 1];
    for j in 1..=m {
        mj[j] = currentw[j - 1];
        mpj[j] = 0;
    }

    for i in 0..=n { ijp[i][0] = i as i32 + 1; }
    for j in 0..=m { ijp[0][j] = -(j as i32 + 1); }

    let mut previousw = vec![0.0f64; m + 1];
    let mut lastverticalw = vec![0.0f64; n + 1];
    if m > 0 { lastverticalw[0] = currentw[m - 1]; }

    let lasti = if tail_gap { n + 1 } else { n };
    for i in 1..lasti {
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[0] = initverticalw[i - 1];

        // match_calc_mtx for row i (C line 1252: uses index i, not i-1)
        // C fills currentw[0..lgth2-1]. Position m is padding (0).
        if i < n {
            for j in 0..m {
                currentw[j] = score_pair(seq1[i], seq2[j]);
            }
        } else {
            // Out-of-bounds tail row: C's over-allocated arrays give zeros
            for j in 0..m {
                currentw[j] = 0.0;
            }
        }
        currentw[m] = 0.0; // padding: no substitution score beyond seq2
        currentw[0] = initverticalw[i];

        // mi = previousw[0], NO ogcp2 multiplication (C line 1275)
        let mut mi = previousw[0];
        let mut mpi: usize = 0;

        // C `Galign11.c:1336-1337`: `fpenalty_ex_i = (i < lgth1) ?
        // fpenalty_ex : 0.0` (boundary fix from 2018/May/11). Match
        // verbatim — at the tail-row (i == n) the in-row extension
        // contribution is zeroed.
        let fpenalty_ex_i = if i < n { penalty_ex } else { 0.0 };
        for j in 1..=m {
            let mut wm = previousw[j - 1];
            ijp[i][j] = 0;

            // Deletion: mi + fpenalty (flat, C line 1306)
            let g = mi + penalty;
            if g > wm {
                wm = g;
                ijp[i][j] = -(j as i32 - mpi as i32);
            }
            // Open from diagonal (C line 1311)
            let g = previousw[j - 1];
            if g >= mi {
                mi = g;
                mpi = j - 1;
            }
            // C `Galign11.c:1362-1363`: `mi += fpenalty_ex_i`
            // — per-cell in-row gap extension contribution.
            mi += fpenalty_ex_i;

            // Insertion: m[j] + fpenalty (flat, C line 1325)
            let g = mj[j] + penalty;
            if g > wm {
                wm = g;
                ijp[i][j] = i as i32 - mpj[j] as i32;
            }
            // Open from diagonal (C line 1330)
            let g = previousw[j - 1];
            if g >= mj[j] {
                mj[j] = g;
                mpj[j] = i - 1;
            }
            // C `Galign11.c:1381-1384`: `if (j < lgth2) m[j] += fpenalty_ex;`
            // — per-cell in-col gap extension contribution, gated on
            // not being the tail column.
            if j < m {
                mj[j] += penalty_ex;
            }

            currentw[j] += wm;
            h[i][j] = currentw[j];
        }
        if m > 0 { lastverticalw[i] = currentw[m - 1]; }
    }

    // Tail gap handling
    if !tail_gap {
        let mut wm = lastverticalw[0];
        for i in 0..n {
            if lastverticalw[i] >= wm {
                wm = lastverticalw[i];
                ijp[n][m] = (n - i) as i32;
            }
        }
        for j in 0..m {
            if h[n - 1][j] >= wm {
                wm = h[n - 1][j];
                ijp[n][m] = -((m - j) as i32);
            }
        }
        // Set h[n][m] to the best score found (for the score field).
        h[n][m] = wm;
    }

    // Traceback (same ijp format as MSalignmm)
    let mut gaptable1 = Vec::new();
    let mut gaptable2 = Vec::new();
    let (mut iin, mut jin) = (n as i32, m as i32);
    let klim = n + m;
    let mut k = 0;
    while k <= klim {
        let v = ijp[iin as usize][jin as usize];
        let (ifi, jfi): (i32, i32) = if v < 0 {
            (iin - 1, jin + v)
        } else if v > 0 {
            (iin - v, jin - 1)
        } else {
            (iin - 1, jin - 1)
        };
        let mut l = iin - ifi;
        while l > 1 { gaptable1.push(b'o'); gaptable2.push(b'-'); k += 1; l -= 1; }
        let mut l = jin - jfi;
        while l > 1 { gaptable1.push(b'-'); gaptable2.push(b'o'); k += 1; l -= 1; }
        if iin <= 0 || jin <= 0 { break; }
        gaptable1.push(b'o'); gaptable2.push(b'o'); k += 1;
        iin = ifi; jin = jfi;
    }

    gaptable1.reverse();
    gaptable2.reverse();
    let mut ops = Vec::with_capacity(gaptable1.len());
    for k in 0..gaptable1.len() {
        match (gaptable1[k], gaptable2[k]) {
            (b'o', b'o') => ops.push(AlignOp::Match),
            (b'o', b'-') => ops.push(AlignOp::Delete),
            (b'-', b'o') => ops.push(AlignOp::Insert),
            _ => {}
        }
    }

    Alignment { seq1: Vec::new(), seq2: Vec::new(), score: h[n][m], operations: ops }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_setup() -> (Vec<Vec<f64>>, [u8; 256]) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 { mtx[i][i] = 100.0; }
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map)
    }

    #[test]
    fn profile_from_single_sequence() {
        let (_, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, 5);
        assert_eq!(prof.length, 4);
        assert!((prof.freqs[0][0] - 1.0).abs() < 1e-10); // A at pos 0
        assert!((prof.freqs[1][1] - 1.0).abs() < 1e-10); // C at pos 1
    }

    #[test]
    fn profile_match_score_identical() {
        let (mtx, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, 5);
        // Score of position 0 vs position 0 should be 100 (A vs A)
        let score = prof.match_score(0, &prof, 0, &mtx);
        assert!((score - 100.0).abs() < 1e-6);
    }

    #[test]
    fn profile_align_identical() {
        let (mtx, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);
        let aln = profile_align(&prof, &prof, &mtx, &gap, true, true);
        assert!((aln.score - 400.0).abs() < 1e-6);
        assert_eq!(aln.operations.len(), 4);
        assert!(aln.operations.iter().all(|op| *op == AlignOp::Match));
    }

    #[test]
    fn profile_with_gaps_reduces_penalty() {
        let (_mtx, map) = simple_setup();
        // Two sequences, one has a gap at position 1
        let seqs: Vec<&[u8]> = vec![b"A-GT", b"ACGT"];
        let prof = Profile::from_aligned(&seqs, &[0.5, 0.5], &map, 5);
        assert!((prof.gap_freq[1] - 0.5).abs() < 1e-10); // 50% gap at pos 1
    }

    #[test]
    fn cpmx_memo_first_call_matches_from_aligned() {
        // Without any prior memo state, from_aligned_with_memo should
        // produce results equivalent to from_aligned (within FP).
        super::reset_cpmx_memo();
        let (_mtx, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGT", b"ACGT", b"AC-T"];
        let w = vec![1.0 / 3.0; 3];
        let p1 = Profile::from_aligned(&seqs, &w, &map, 5);
        let p2 = Profile::from_aligned_with_memo(&seqs, &w, &map, 5, 0, 3, 4);
        // FROM-SCRATCH path → bit-identical to from_aligned.
        for pos in 0..4 {
            for k in 0..5 {
                let a = p1.freqs[pos][k];
                let b = p2.freqs[pos][k];
                assert_eq!(a, b, "freqs[{}][{}]: from_aligned={a:e} memo={b:e}", pos, k);
            }
        }
        for pos in 0..4 {
            assert_eq!(p1.gap_freq[pos], p2.gap_freq[pos]);
            assert_eq!(p1.ogcp[pos], p2.ogcp[pos]);
            assert_eq!(p1.fgcp[pos], p2.fgcp[pos]);
        }
    }

    #[test]
    fn cpmx_memo_incremental_matches_from_scratch_for_extension() {
        // When the memo activates (extension by one sequence), the
        // incremental and from-scratch paths should produce the same
        // mathematical result (modulo 1-ULP FP precision).
        super::reset_cpmx_memo();
        let (_mtx, map) = simple_setup();
        let two_seqs: Vec<&[u8]> = vec![b"ACGT", b"AC-T"];
        let w2 = vec![0.5, 0.5];
        // First call: cluster of 2 sequences.
        let _p2 = Profile::from_aligned_with_memo(
            &two_seqs, &w2, &map, 5, 0 /*firstmem*/, 2 /*icyc*/, 4 /*lgth*/);
        // Second call: cluster of 3 sequences (extends with one more).
        // C's effective normalized weights are 1/3 each. In incremental
        // form (per cpmx_calc_add): new cpmx = old * (1 - 1/3) + 1/3 * delta_new.
        // But we passed prev weights as [0.5, 0.5], so the existing cpmx
        // values match a 2-way profile. To extend, the caller normally
        // re-normalizes weights to sum to 1.0 within the new cluster.
        let three_seqs: Vec<&[u8]> = vec![b"ACGT", b"AC-T", b"AGGT"];
        let w3 = vec![1.0 / 3.0; 3];
        let p3_incr = Profile::from_aligned_with_memo(
            &three_seqs, &w3, &map, 5, 0, 3, 4);
        // Compare to from-scratch on the 3-way cluster.
        let p3_scratch = Profile::from_aligned(&three_seqs, &w3, &map, 5);
        // Note: incremental path uses the old cpmx (built from 2 seqs
        // with weights [0.5,0.5]) scaled by (1 - 1/3) = 2/3, then adds
        // 1/3 * delta_new. The scaled-old equals from-scratch on 2 seqs
        // with weights [0.5*2/3, 0.5*2/3] = [1/3, 1/3]. Plus 1/3 for
        // the 3rd seq. Net: matches from-scratch on 3 seqs with [1/3,
        // 1/3, 1/3]. Mathematically equal; FP-differently arrived at.
        let mut max_diff: f64 = 0.0;
        for pos in 0..4 {
            for k in 0..5 {
                let d = (p3_incr.freqs[pos][k] - p3_scratch.freqs[pos][k]).abs();
                max_diff = max_diff.max(d);
            }
        }
        // Allow up to a few ULP — the incremental path has different
        // FP rounding than from-scratch.
        assert!(max_diff < 1e-14,
            "incremental vs from-scratch diverge by {:.3e}", max_diff);
    }
}
