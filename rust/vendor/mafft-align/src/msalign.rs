//! Hirschberg-style linear-space profile DP — port of C's `MSalignmm`
//! (`mafft-upstream/core/MSalignmm.c`).
//!
//! C MAFFT routes `--memsave` (`alg = 'M'`, `tbfast.c:1159-1161`) and
//! the auto-switch path at `len > 30000` (`tbfast.c:1096-1104`) through
//! `MSalignmm`. The DP recurrence is identical to `A__align` /
//! [`profile_align_imp_with_boundary`]; the only difference is memory
//! layout — `A__align` allocates an O(N×M) trace matrix, `MSalignmm`
//! uses Hirschberg's divide-and-conquer to work in O(N+M).
//!
//! Implementation strategy:
//!   * Base case (`lgth1 < DPTANNI=100` OR `lgth2 < DPTANNI`): delegate
//!     to [`profile_align_imp_with_boundary`]. The full DP is fine for
//!     small sub-problems.
//!   * Recursive case: forward DP from row `ist` to row `imid =
//!     ist + lgth1/2`, capturing `midw[j]`, `midm[j]`, `midn[j]` for
//!     each column j at the midpoint row. Backward DP from row `ien`
//!     to `imid`, adding to those arrays. Find `jmid` = argmax over
//!     midw/midm/midn. Recurse on top-left `(ist, ist+jumpi, jst,
//!     jst+jumpj)` and bottom-right `(ist+imid, ien, jst+jmid, jen)`
//!     with appropriate inter-half gap padding.
//!
//! ## Correctness scope
//!
//! Verified:
//!   * Base case (< DPTANNI on either axis) is byte-identical to
//!     [`profile_align_imp_with_boundary`], because the recursion
//!     bottoms out directly into it.
//!   * Recursive case returns the SAME optimal score as the full DP
//!     for every test input — both for identical-input traces (no
//!     ties) and for gap-required traces with unique optima.
//!   * For inputs with a unique optimal alignment, the trace is
//!     byte-identical to the full DP.
//!
//! Known residual (tracked in TODO §B.3):
//!   * For inputs with SCORE-TIED optimal alignments (multiple traces
//!     of equal score), the trace selection in the Hirschberg
//!     forward/backward midpoint may pick a different equally-scored
//!     trace than C `MSalignmm`. This does not affect the alignment
//!     score, only which residue columns the gaps land in. Matching
//!     C MSalignmm's exact tie-break behavior requires per-step
//!     cross-validation against the C implementation; not yet wired.
//!
//! Because of the tie-break residual, this module is currently a
//! library-level entry point: `--memsave` continues to flow through
//! the full DP path (which is byte-identical to C MSalignmm for our
//! test inputs). Use [`msalignmm`] explicitly when you want
//! linear-space DP — e.g. for sequences > 30000 in length where the
//! full DP would OOM.

use crate::dp::{Alignment, AlignOp, GapModel};
use crate::profile::{Profile, profile_align_imp_with_boundary, BoundaryFreqs};

/// C `MSalignmm.c:11` — sub-problem size threshold below which the
/// recursion bottoms out into the direct DP.
const DPTANNI: usize = 100;

/// Top-level entry point. Mirrors C `MSalignmm` (line 2094): set up
/// `headgapfreq1/2` defaults (= 1.0, since the `sgap1`/`egap1` path
/// isn't relevant for `--memsave`), then drive `msalignmm_rec`.
pub fn msalignmm(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
) -> Alignment {
    let n = prof1.length;
    let m = prof2.length;
    if n == 0 || m == 0 {
        return Alignment {
            seq1: Vec::new(),
            seq2: Vec::new(),
            score: 0.0,
            operations: Vec::new(),
        };
    }
    let mut ops = Vec::with_capacity(n + m);
    let score = msalignmm_rec(
        prof1, prof2, matrix, gap,
        0, n - 1, 0, m - 1,
        head_gap, tail_gap,
        1.0, 1.0,
        &mut ops,
    );
    Alignment {
        seq1: Vec::new(),
        seq2: Vec::new(),
        score,
        operations: ops,
    }
}

/// Recursive Hirschberg DP. `ist`/`ien` and `jst`/`jen` are inclusive
/// absolute positions into the parent `prof1` / `prof2`. The function
/// appends the trace operations for the sub-region into `out_ops` and
/// returns the score for that sub-region.
///
/// Mirrors C `MSalignmm_rec` (line 991). The `headgapfreq{1,2}_g`
/// parameters carry the "nongap fraction at position ist-1 / jst-1"
/// when those positions don't exist in the parent profile (i.e. for
/// the top-level call); for nested calls the actual previous column's
/// nongap_freq is used (matching C line 1072-1075).
fn msalignmm_rec(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    ist: usize, ien: usize,
    jst: usize, jen: usize,
    head_gap: bool, tail_gap: bool,
    headgapfreq1_g: f64, headgapfreq2_g: f64,
    out_ops: &mut Vec<AlignOp>,
) -> f64 {
    let lgth1 = ien - ist + 1;
    let lgth2 = jen - jst + 1;

    // Base case: small sub-problem → full DP.
    // C `MSalignmm_rec:1144` — `if( lgth1 < DPTANNI || lgth2 < DPTANNI )`.
    if lgth1 < DPTANNI || lgth2 < DPTANNI {
        return base_case(
            prof1, prof2, matrix, gap,
            ist, ien, jst, jen,
            head_gap, tail_gap,
            headgapfreq1_g, headgapfreq2_g,
            out_ops,
        );
    }

    // C `MSalignmm_rec:1261` — `imid = lgth1 * 0.5;` (integer truncation).
    let imid = lgth1 / 2;

    // Forward DP rows 1..=imid, accumulating midw/midm/midn at row imid.
    let fwd = forward_dp(
        prof1, prof2, matrix, gap,
        ist, ien, jst, jen,
        head_gap,
        imid,
        headgapfreq1_g, headgapfreq2_g,
    );

    // Backward DP rows lgth1-2 down to imid-1, adding to midw/midm/midn.
    let bwd = backward_dp(
        prof1, prof2, matrix, gap,
        ist, ien, jst, jen,
        tail_gap,
        imid,
        fwd,
    );

    let (jumpi, jumpj, jmid) = bwd.split_point;
    let imid = bwd.imid_override.unwrap_or(imid);

    // Recurse on top-left sub-region `(ist .. ist+jumpi, jst .. jst+jumpj)`.
    // C `MSalignmm_rec:1886` — both endpoints inclusive. For a diagonal
    // join (`midw` winner), `jumpi = imid-1` and `jumpj = jmid-1`, so
    // top covers `[ist, ist+imid-1]` (imid rows). Bottom-right starts
    // at `ist+imid` so the union is exact.
    let mut value = 0.0;
    value += msalignmm_rec(
        prof1, prof2, matrix, gap,
        ist, ist + jumpi, jst, jst + jumpj,
        head_gap, false,
        headgapfreq1_g, headgapfreq2_g,
        out_ops,
    );

    // Inter-half gap padding for prof1: `l = jmid - jumpj - 1`
    // (C line 1907). This is a run of gap-in-prof1 / residue-in-prof2.
    let l_horiz: i64 = jmid as i64 - jumpj as i64 - 1;
    if l_horiz > 0 {
        for _ in 0..l_horiz {
            out_ops.push(AlignOp::Insert);
        }
        // Gap-cost contribution (C line 1925):
        //   value += ogcp2[jumpj+1] + fgcp2[jmid-1]
        // In C, `ogcp2 = gapinfo[2] + jst` is a SLICE starting at jst,
        // so `ogcp2[jumpj+1]` is `ogcp2_full[jst + jumpj + 1]` in
        // absolute coordinates. Same for fgcp2.
        let ogcp2 = effective_ogcp2(prof2, gap);
        let fgcp2 = effective_fgcp2(prof2, gap);
        let idx_o = jst + jumpj + 1;
        let idx_f = jst + jmid - 1;
        if idx_o < ogcp2.len() && idx_f < fgcp2.len() {
            value += ogcp2[idx_o] + fgcp2[idx_f];
        }
    }
    // Inter-half gap padding for prof2: `l = imid - jumpi - 1`
    // (C line 1929).
    let l_vert: i64 = imid as i64 - jumpi as i64 - 1;
    if l_vert > 0 {
        for _ in 0..l_vert {
            out_ops.push(AlignOp::Delete);
        }
        // C line 1953:  value += ogcp1[jumpi+1] + fgcp1[imid-1]
        // — slice-relative, same +1 / -1 convention as prof2.
        let ogcp1 = effective_ogcp1(prof1, gap);
        let fgcp1 = effective_fgcp1(prof1, gap);
        let idx_o = ist + jumpi + 1;
        let idx_f = ist + imid - 1;
        if idx_o < ogcp1.len() && idx_f < fgcp1.len() {
            value += ogcp1[idx_o] + fgcp1[idx_f];
        }
    }

    // Recurse on bottom-right sub-region `(ist+imid .. ien, jst+jmid .. jen)`.
    // For the bottom recursion the head boundary lives INSIDE the
    // parent: C `MSalignmm_tanni` derives `headgapfreq1 = gapfreq1f[-1]`
    // from the original profile when `ist > 0` (line 709-712). We
    // mirror that here by passing the previous-column nongap_freq.
    let bottom_ist = ist + imid;
    let bottom_jst = jst + jmid;
    if bottom_ist <= ien && bottom_jst <= jen {
        let head1_for_bot = if bottom_ist > 0 && bottom_ist - 1 < prof1.length {
            prof1.nongap_freq[bottom_ist - 1]
        } else { 1.0 };
        let head2_for_bot = if bottom_jst > 0 && bottom_jst - 1 < prof2.length {
            prof2.nongap_freq[bottom_jst - 1]
        } else { 1.0 };
        value += msalignmm_rec(
            prof1, prof2, matrix, gap,
            bottom_ist, ien, bottom_jst, jen,
            false, tail_gap,
            head1_for_bot, head2_for_bot,
            out_ops,
        );
    }

    value
}

/// Base case: sub-region small enough that the full O(lgth1 × lgth2)
/// trace matrix fits comfortably. Use the existing
/// [`profile_align_imp_with_boundary`] (= our port of C `A__align` /
/// `MSalignmm_tanni` — they share the same recurrence).
fn base_case(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    ist: usize, ien: usize,
    jst: usize, jen: usize,
    head_gap: bool, tail_gap: bool,
    headgapfreq1_g: f64, headgapfreq2_g: f64,
    out_ops: &mut Vec<AlignOp>,
) -> f64 {
    let sub1 = prof1.sub_profile(ist, ien + 1);
    let sub2 = prof2.sub_profile(jst, jen + 1);

    // Derive boundary nongap fractions from the parent profile.
    // C `MSalignmm_tanni:709-712`:
    //   if( ist > 0 ) headgapfreq1 = gapfreq1f[-1];
    //   else headgapfreq1 = headgapfreq1_g;
    // (and similarly for tail). `gapfreq1f` is `1 - gap_freq` per
    // position (the "nongap fraction"); `gapfreq1f[-1]` peeks at
    // position `ist - 1` in the parent profile.
    let head1 = if ist > 0 { prof1.nongap_freq[ist - 1] } else { headgapfreq1_g };
    let head2 = if jst > 0 { prof2.nongap_freq[jst - 1] } else { headgapfreq2_g };
    let tail1 = if ien + 1 < prof1.length { prof1.nongap_freq[ien + 1] } else { 1.0 };
    let tail2 = if jen + 1 < prof2.length { prof2.nongap_freq[jen + 1] } else { 1.0 };

    // Internal sub-regions always treat head/tail as "open" (the gap
    // padding between halves is added in the recursion glue, not in
    // the base case). This matches `MSalignmm_tanni`'s
    // `if( headgp || ist != 0 )` and `if( tailgp || jen != fulllen2-1 )`
    // — when `ist != 0`, the head_gap branch fires regardless of
    // `headgp` because the boundary is internal.
    let effective_head = head_gap || ist != 0 || jst != 0;
    let effective_tail = tail_gap || ien + 1 != prof1.length || jen + 1 != prof2.length;

    let aln = profile_align_imp_with_boundary(
        &sub1, &sub2, matrix, gap, effective_head, effective_tail, None, false,
        BoundaryFreqs { head1, head2, tail1, tail2 },
    );
    out_ops.extend(aln.operations);
    aln.score
}

/// Result of the forward DP up to and including row `imid`.
struct ForwardState {
    /// `midw[j]` after the forward pass at row `imid`:
    /// `currentw[j] + match_calc` accumulated up to row `imid`.
    midw: Vec<f64>,
    /// `midm[j]` after the forward pass: the row-running `m[j]` value
    /// (gap-extend tracker in prof2 direction) at row `imid`.
    midm: Vec<f64>,
    /// `midn[j]` after the forward pass: the col-running `mi` value
    /// (gap-extend tracker in prof1 direction) at row `imid`.
    midn: Vec<f64>,
    /// Trace metadata for the forward sweep at row `imid`:
    /// `jumpbackj[j] = *mpjpt` at the column update (i.e. the prior i
    /// where mj[j] was last raised), `jumpbacki[j] = mpi` (the prior j
    /// where mi was last raised).
    jumpbackj: Vec<i64>,
    jumpbacki: Vec<i64>,
}

/// Result of the backward DP plus the split-point selection.
struct BackwardState {
    /// `(jumpi, jumpj, jmid)` — the optimal split coordinates.
    /// `jumpi`/`jumpj` are the i/j right BEFORE the cut (top-left
    /// half ends at `ist+jumpi, jst+jumpj` inclusive); `imid`/`jmid`
    /// are the i/j right AFTER the cut (bottom-right half starts at
    /// `ist+imid, jst+jmid`).
    split_point: (usize, usize, usize),
    /// If `Some`, overrides the parent's `imid = lgth1/2` with
    /// `jumpforwi[jumpj]` (C line 1767). For typical splits this
    /// equals `imid` (no change); for splits where the mj-branch
    /// fed `ijpi = mp[dj]` it points further down, shifting the
    /// bottom-half recursion start row.
    imid_override: Option<usize>,
}

/// Forward DP from row `ist` to row `ist + imid` (inclusive of the
/// midpoint sweep). Mirrors C `MSalignmm_rec:1220-1413` — the same
/// recurrence as `MSalignmm_tanni` / [`profile_align_imp_with_boundary`]
/// but without the O(N×M) trace matrix.
fn forward_dp(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    ist: usize, _ien: usize,
    jst: usize, jen: usize,
    head_gap: bool,
    imid: usize,
    headgapfreq1_g: f64, headgapfreq2_g: f64,
) -> ForwardState {
    let lgth2 = jen - jst + 1;
    let nalpha = prof1.nalphabets.min(prof2.nalphabets).min(matrix.len());

    let ogcp1 = effective_ogcp1(prof1, gap);
    let fgcp1 = effective_fgcp1(prof1, gap);
    let ogcp2 = effective_ogcp2(prof2, gap);
    let fgcp2 = effective_fgcp2(prof2, gap);

    // Use the parent profile's nongap_freq for `gapfreq{1,2}f` and the
    // headgapfreq{1,2}_g for the absolute boundary (matching C
    // MSalignmm_rec:1072-1075).
    let headgapfreq1 = if ist > 0 { prof1.nongap_freq[ist - 1] } else { headgapfreq1_g };
    let headgapfreq2 = if jst > 0 { prof2.nongap_freq[jst - 1] } else { headgapfreq2_g };

    let _ = gap.extend; // C MSalignmm has USE_PENALTY_EX = 0; f_ext is unused.

    // `initverticalw`: `match_calc(prof2[jst], prof1[ist..])` then add
    // `ogcp1[ist] * headgapfreq2 + fgcp1[ist+i-1] * nongap_freq2[jst]`.
    // We collapse "size lgth1+1" to "absolute indices ist..=ien" — the
    // first element corresponds to row 0 of the sub-region.
    let mut initverticalw = vec![0.0f64; imid + 2];
    {
        // match_calc(initverticalw, cpmx2[jst], cpmx1[ist..]).
        let mut scarr = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            let mut s = 0.0f64;
            for k in 0..nalpha {
                s = matrix[k][l].mul_add(prof2.freqs[jst][k], s);
            }
            scarr[l] = s;
        }
        for di in 0..=imid {
            let row = ist + di;
            if row >= prof1.length { break; }
            let mut s = 0.0f64;
            for l in 0..nalpha {
                s = scarr[l].mul_add(prof1.freqs[row][l], s);
            }
            initverticalw[di] = s;
        }
        // Add head-gap penalty (C line 1229-1233). C MSalignmm has
        // `USE_PENALTY_EX = 0` (MSalignmm.c:7), so the `+= fpenalty_ex
        // * i` increment that `A__align` (`profile_align_imp_with_
        // boundary`) applies in its head-gap init is OMITTED here.
        let gf2_0 = prof2.nongap_freq[jst];
        if head_gap || ist != 0 {
            for di in 1..=imid {
                if di >= initverticalw.len() { break; }
                let row = ist + di - 1;
                if row >= prof1.length { break; }
                initverticalw[di] = fgcp1[row].mul_add(
                    gf2_0,
                    ogcp1[ist].mul_add(headgapfreq2, initverticalw[di]),
                );
            }
        }
    }

    // `currentw`: `match_calc(prof1[ist], prof2[jst..])` then add
    // head-gap penalty for prof2 axis.
    let mut currentw = vec![0.0f64; lgth2 + 1];
    let mut previousw = vec![0.0f64; lgth2 + 1];
    {
        let mut scarr = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            let mut s = 0.0f64;
            for k in 0..nalpha {
                s = matrix[k][l].mul_add(prof1.freqs[ist][k], s);
            }
            scarr[l] = s;
        }
        for dj in 0..lgth2 {
            let col = jst + dj;
            if col >= prof2.length { break; }
            let mut s = 0.0f64;
            for l in 0..nalpha {
                s = scarr[l].mul_add(prof2.freqs[col][l], s);
            }
            currentw[dj] = s;
        }
        let gf1_0 = prof1.nongap_freq[ist];
        if head_gap || jst != 0 {
            for dj in 1..=lgth2 {
                if dj >= currentw.len() { break; }
                let col = jst + dj - 1;
                if col >= prof2.length { break; }
                currentw[dj] = fgcp2[col].mul_add(
                    gf1_0,
                    ogcp2[jst].mul_add(headgapfreq1, currentw[dj]),
                );
            }
        }
    }

    // `m[j] = currentw[j-1] + ogcp1[ist+1] * nongap_freq2[jst+j-1]`
    // C line 1252-1257.
    let mut m = vec![0.0f64; lgth2 + 1];
    let mut mp = vec![0i64; lgth2 + 1];
    for dj in 1..=lgth2 {
        let col_prev = jst + dj - 1;
        let gf2_prev = if col_prev < prof2.length { prof2.nongap_freq[col_prev] } else { 1.0 };
        let row_o = (ist + 1).min(prof1.length.saturating_sub(1));
        m[dj] = ogcp1[row_o].mul_add(gf2_prev, currentw[dj - 1]);
        mp[dj] = 0;
    }

    // Storage for the midpoint row state.
    let mut midw = vec![0.0f64; lgth2 + 1];
    let mut midm = vec![0.0f64; lgth2 + 1];
    let mut midn = vec![0.0f64; lgth2 + 1];
    let mut jumpbackj = vec![0i64; lgth2 + 1];
    let mut jumpbacki = vec![0i64; lgth2 + 1];

    // Forward DP rows 1..=imid (relative to ist).
    // Mirrors C `MSalignmm_rec` inner loop (lines 1268-1413).
    for di in 1..=imid {
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[0] = initverticalw[di - 1];

        let row = ist + di;
        if row >= prof1.length {
            for v in currentw.iter_mut() { *v = 0.0; }
        } else {
            let mut scarr = vec![0.0f64; nalpha];
            for l in 0..nalpha {
                let mut s = 0.0f64;
                for k in 0..nalpha {
                    s = matrix[k][l].mul_add(prof1.freqs[row][k], s);
                }
                scarr[l] = s;
            }
            for dj in 0..lgth2 {
                let col = jst + dj;
                if col >= prof2.length {
                    currentw[dj] = 0.0;
                    continue;
                }
                let mut s = 0.0f64;
                for l in 0..nalpha {
                    s = scarr[l].mul_add(prof2.freqs[col][l], s);
                }
                currentw[dj] = s;
            }
        }
        currentw[0] = initverticalw[di];
        // C line 1301: `m[0] = ogcp1[i]` — for absolute index ist+di.
        let row_idx = (ist + di).min(prof1.length.saturating_sub(1));
        m[0] = ogcp1[row_idx];

        // `mi = previousw[0] + ogcp2[jst+1] * nongap_freq1[ist+di-1]`
        let row_prev = ist + di - 1;
        let gf1_prev = if row_prev < prof1.length { prof1.nongap_freq[row_prev] } else { 1.0 };
        let col_o = (jst + 1).min(prof2.length.saturating_sub(1));
        let mut mi = ogcp2[col_o].mul_add(gf1_prev, previousw[0]);
        let mut mpi: i64 = 0;

        // Capture m[0] at midpoint
        if di == imid { midm[0] = m[0]; }

        for dj in 1..=lgth2 {
            let row_i = ist + di;
            let row_im1 = ist + di - 1;
            let col_j = jst + dj;
            let col_jm1 = jst + dj - 1;

            let gf1_i = if row_i < prof1.length {
                prof1.nongap_freq[row_i]
            } else { 1.0 };
            let gf1_im1 = if row_im1 < prof1.length {
                prof1.nongap_freq[row_im1]
            } else { 1.0 };
            let gf2_j = if col_j < prof2.length {
                prof2.nongap_freq[col_j]
            } else { 1.0 };
            let gf2_jm1 = if col_jm1 < prof2.length {
                prof2.nongap_freq[col_jm1]
            } else { 1.0 };

            let mut wm = previousw[dj - 1];

            // mi (row-running gap-skip tracker)
            // C line 1328: g = mi + fgcp2[col_jm1] * gf1_i
            let g = fgcp2[col_jm1].mul_add(gf1_i, mi);
            if g > wm { wm = g; }
            // C line 1338: g = previousw[dj-1] + ogcp2[col_j] * gf1_im1
            let g = ogcp2[col_j].mul_add(gf1_im1, previousw[dj - 1]);
            if g >= mi {
                mi = g;
                mpi = (dj - 1) as i64;
            }
            // C MSalignmm has `USE_PENALTY_EX = 0` (MSalignmm.c:7), so
            // the `mi += fpenalty_ex;` increment that `A__align`
            // applies is OMITTED here.

            // m[dj] (column-running gap-skip tracker)
            // C line 1349: g = m[dj] + fgcp1[row_im1] * gf2_j
            let g = fgcp1[row_im1].mul_add(gf2_j, m[dj]);
            if g > wm { wm = g; }
            // C line 1361: g = previousw[dj-1] + ogcp1[row_i] * gf2_jm1
            let g = ogcp1[row_i].mul_add(gf2_jm1, previousw[dj - 1]);
            if g >= m[dj] {
                m[dj] = g;
                mp[dj] = (di - 1) as i64;
            }
            // `USE_PENALTY_EX = 0`: skip `m[dj] += fpenalty_ex;` too.

            currentw[dj] += wm;

            // At the midpoint, capture state for the join with backward.
            if di == imid {
                jumpbackj[dj] = mp[dj];
                jumpbacki[dj] = mpi;
                midw[dj] = currentw[dj];
                midm[dj] = m[dj];
                midn[dj] = mi;
            }
        }
    }

    ForwardState { midw, midm, midn, jumpbackj, jumpbacki }
}

/// Backward DP from row `ien` (= `ist + lgth1 - 1`) down to `imid - 1`.
/// Adds the partial scores to `fwd.midw / midm / midn` at the
/// midpoint, then finds the optimal split column. Mirrors C
/// `MSalignmm_rec:1442-1785`.
fn backward_dp(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    ist: usize, ien: usize,
    jst: usize, jen: usize,
    tail_gap: bool,
    imid: usize,
    fwd: ForwardState,
) -> BackwardState {
    let lgth1 = ien - ist + 1;
    let lgth2 = jen - jst + 1;
    let nalpha = prof1.nalphabets.min(prof2.nalphabets).min(matrix.len());

    let ogcp1 = effective_ogcp1(prof1, gap);
    let fgcp1 = effective_fgcp1(prof1, gap);
    let ogcp2 = effective_ogcp2(prof2, gap);
    let fgcp2 = effective_fgcp2(prof2, gap);

    let _ = gap.extend; // C MSalignmm has USE_PENALTY_EX = 0; f_ext is unused.

    let tail1: f64 = if ien + 1 < prof1.length { prof1.nongap_freq[ien + 1] } else { 1.0 };
    let tail2: f64 = if jen + 1 < prof2.length { prof2.nongap_freq[jen + 1] } else { 1.0 };

    let _ = tail_gap; // tail_gap is implicitly handled via tail1/tail2 boundaries

    // initverticalw[di] = match_calc(prof2[jen], prof1[ist..=ien]) +
    //   fgcp1[ien] * gf2_lgth2 + ogcp1[ist+di+1] * gf2_lgth2m1
    let mut initverticalw = vec![0.0f64; lgth1 + 1];
    {
        let mut scarr = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            let mut s = 0.0f64;
            for k in 0..nalpha {
                s = matrix[k][l].mul_add(prof2.freqs[jen][k], s);
            }
            scarr[l] = s;
        }
        for di in 0..lgth1 {
            let row = ist + di;
            if row >= prof1.length { break; }
            let mut s = 0.0f64;
            for l in 0..nalpha {
                s = scarr[l].mul_add(prof1.freqs[row][l], s);
            }
            initverticalw[di] = s;
        }
        let gf2_tail = tail2;
        let gf2_lgth2m1 = if jen < prof2.length { prof2.nongap_freq[jen] } else { 1.0 };
        for di in 0..lgth1.saturating_sub(1) {
            initverticalw[di] = fgcp1[ien].mul_add(
                gf2_tail,
                ogcp1[ist + di + 1].mul_add(gf2_lgth2m1, initverticalw[di]),
            );
        }
    }

    // currentw[dj] = match_calc(prof1[ien], prof2[jst..=jen]) +
    //   fgcp2[jen] * gf1_lgth1 + ogcp2[jst+dj+1] * gf1_lgth1m1
    let mut currentw = vec![0.0f64; lgth2 + 1];
    let mut previousw = vec![0.0f64; lgth2 + 1];
    {
        let mut scarr = vec![0.0f64; nalpha];
        for l in 0..nalpha {
            let mut s = 0.0f64;
            for k in 0..nalpha {
                s = matrix[k][l].mul_add(prof1.freqs[ien][k], s);
            }
            scarr[l] = s;
        }
        for dj in 0..lgth2 {
            let col = jst + dj;
            if col >= prof2.length { break; }
            let mut s = 0.0f64;
            for l in 0..nalpha {
                s = scarr[l].mul_add(prof2.freqs[col][l], s);
            }
            currentw[dj] = s;
        }
        let gf1_tail = tail1;
        let gf1_lgth1m1 = if ien < prof1.length { prof1.nongap_freq[ien] } else { 1.0 };
        for dj in 0..lgth2.saturating_sub(1) {
            currentw[dj] = fgcp2[jen].mul_add(
                gf1_tail,
                ogcp2[jst + dj + 1].mul_add(gf1_lgth1m1, currentw[dj]),
            );
        }
    }

    // m[dj] = currentw[dj+1] + fgcp1[ien-1] * gf2_{jen-dj+1}
    let mut m = vec![0.0f64; lgth2 + 1];
    let mut mp = vec![0i64; lgth2 + 1];
    let row_end_prev = ien.saturating_sub(1);
    for dj in (0..lgth2).rev() {
        let col_next = jst + dj + 1;
        let gf2_next = if col_next < prof2.length { prof2.nongap_freq[col_next] } else { 1.0 };
        let cw = if dj + 1 < currentw.len() { currentw[dj + 1] } else { 0.0 };
        m[dj] = fgcp1[row_end_prev].mul_add(gf2_next, cw);
        mp[dj] = (lgth1 as i64) - 1;
    }

    // Storage for split-point detection.
    let mut jmid: usize = 0;
    let mut jumpi: usize = imid.saturating_sub(1);
    let mut jumpj: usize = 0;
    let mut jumpforwi = vec![0i64; lgth2 + 1];
    let mut jumpforwj = vec![0i64; lgth2 + 1];
    let mut firstm: f64 = f64::NEG_INFINITY;
    let mut firstmp: i64 = lgth1 as i64;

    let mut midw = fwd.midw;
    let mut midm = fwd.midm;
    let mut midn = fwd.midn;
    let jumpbacki = fwd.jumpbacki;
    let jumpbackj = fwd.jumpbackj;

    // Backward DP rows lgth1-2 .. imid-1 (relative to ist). We break
    // at i == imid - 1, immediately after the split decision —
    // matching C's behaviour for w/n-splits. For m-splits C continues
    // down to i = jumpbackj[jmid] and refreshes jumpforwi/jumpforwj
    // (C line 1597-1603); we don't yet, so the trace for asymmetric
    // inputs with m-split-optimal alignments may diverge from C
    // (TODO §B.3).
    let mut di_signed: i64 = (lgth1 as i64) - 2;
    while di_signed >= 0 {
        let di = di_signed as usize;
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[lgth2 - 1] = initverticalw[di + 1];

        let row = ist + di;
        if row >= prof1.length {
            for v in currentw.iter_mut() { *v = 0.0; }
        } else {
            let mut scarr = vec![0.0f64; nalpha];
            for l in 0..nalpha {
                let mut s = 0.0f64;
                for k in 0..nalpha {
                    s = matrix[k][l].mul_add(prof1.freqs[row][k], s);
                }
                scarr[l] = s;
            }
            for dj in 0..lgth2 {
                let col = jst + dj;
                if col >= prof2.length {
                    currentw[dj] = 0.0;
                    continue;
                }
                let mut s = 0.0f64;
                for l in 0..nalpha {
                    s = scarr[l].mul_add(prof2.freqs[col][l], s);
                }
                currentw[dj] = s;
            }
        }
        currentw[lgth2 - 1] = initverticalw[di];

        let row_i = ist + di;
        let row_ip1 = ist + di + 1;
        let gf1_ip1 = if row_ip1 < prof1.length { prof1.nongap_freq[row_ip1] } else { 1.0 };
        let col_end_prev_local = lgth2 - 2;
        let col_end_prev_abs = jst + col_end_prev_local;
        let mut mi = fgcp2[col_end_prev_abs].mul_add(gf1_ip1, previousw[lgth2 - 1]);
        let mut mpi: i64 = (lgth2 - 1) as i64;

        let mut dj_signed: i64 = (lgth2 as i64) - 2;
        while dj_signed >= 0 {
            let dj = dj_signed as usize;
            let col_j = jst + dj;
            let col_jp1 = jst + dj + 1;
            let gf1_i = if row_i < prof1.length { prof1.nongap_freq[row_i] } else { 1.0 };
            let gf1_ip1 = if row_ip1 < prof1.length { prof1.nongap_freq[row_ip1] } else { 1.0 };
            let gf2_j = if col_j < prof2.length { prof2.nongap_freq[col_j] } else { 1.0 };
            let gf2_jp1 = if col_jp1 < prof2.length { prof2.nongap_freq[col_jp1] } else { 1.0 };

            // Diagonal default (C lines 1548-1550). In the C
            // MSalignmm backward inner loop, `*prept = previousw[j+1]`
            // because `prept` starts at `previousw + lgth2 - 1` and
            // decrements (so at iter j=lgth2-2, *prept = previousw[lgth2-1]
            // = previousw[(lgth2-2) + 1]). All four `*prept` references
            // below therefore use `previousw[dj + 1]`, NOT `previousw[dj]`.
            let pprev = previousw[dj + 1];
            let mut wm = pprev;
            let mut ijpi: i64 = (di + 1) as i64;
            let mut ijpj: i64 = (dj + 1) as i64;

            // C line 1552 (mi candidate):
            //   g = mi + ogcp2[col_jp1] * gf1_i
            //   if g > wm: wm = g, ijpj = mpi, ijpi = i+1
            let g = ogcp2[col_jp1].mul_add(gf1_i, mi);
            if g > wm {
                wm = g;
                ijpj = mpi;
                ijpi = (di + 1) as i64;
            }

            // mi update (C line 1561). g = *prept + fgcp2[j] * gf1[i+1]
            //   = previousw[dj+1] + fgcp2[col_j] * gf1_ip1.
            let g = fgcp2[col_j].mul_add(gf1_ip1, pprev);
            if g >= mi {
                mi = g;
                mpi = (dj + 1) as i64;
            }
            // C MSalignmm has `USE_PENALTY_EX = 0`: skip `mi += fpenalty_ex`.

            // C line 1575 (mj candidate):
            //   g = m[dj] + ogcp1[row_ip1] * gf2_j
            //   if g > wm: wm = g, ijpi = mp[dj], ijpj = j+1
            let g = ogcp1[row_ip1].mul_add(gf2_j, m[dj]);
            if g > wm {
                wm = g;
                ijpi = mp[dj];
                ijpj = (dj + 1) as i64;
            }

            // mj update (C line 1585). g = *prept + fgcp1[i] * gf2_{j+1}
            //   = previousw[dj+1] + fgcp1[row_i] * gf2_jp1.
            let g = fgcp1[row_i].mul_add(gf2_jp1, pprev);
            if g >= m[dj] {
                m[dj] = g;
                mp[dj] = (di + 1) as i64;
            }
            // `USE_PENALTY_EX = 0`: skip `m[dj] += fpenalty_ex`.

            // jumpforwi/jumpforwj writes at i == imid - 1 (the only
            // case we currently handle — see TODO §B.3 m-split note).
            if di == imid.saturating_sub(1) {
                jumpforwi[dj] = ijpi;
                jumpforwj[dj] = ijpj;
            }
            // Accumulate midw/midm at row imid; midn at row imid-1.
            // C `MSalignmm.c:1610-1612`:
            //   midw[j] += wm;        // NOTE: j, not j+1
            //   midm[j+1] += *mjpt;
            // The midw index off-by-one (j vs j+1) was a port bug
            // that surfaced as a 1-column drift on asymmetric inputs.
            if di == imid {
                if dj < midw.len() { midw[dj] += wm; }
                if dj + 1 < midm.len() { midm[dj + 1] += m[dj]; }
            }
            if di == imid.saturating_sub(1) {
                if dj < midn.len() { midn[dj] += mi; }
            }

            currentw[dj] += wm;

            dj_signed -= 1;
        }
        // C line 1628: track firstm = max over rows of previousw[0] + fgcp1[i].
        let g_first = fgcp1[row_i] + previousw[0];
        if firstm < g_first {
            firstm = g_first;
            firstmp = (di + 1) as i64;
        }
        // C line 1637: `if( i == imid ) midm[j+1] += firstm;` — at
        // this point the inner-loop j has decremented past 0 to -1,
        // so `j + 1` is 0. C touches `midm[0]`.
        if di == imid {
            if !midm.is_empty() { midm[0] += firstm; }
        }

        // At i == imid - 1, decide jmid + (jumpi, jumpj) and break.
        if di == imid.saturating_sub(1) {
            let mut maxwm = midw.get(1).copied().unwrap_or(f64::NEG_INFINITY);
            jmid = 0;
            for j in 2..lgth2.saturating_sub(1) {
                if let Some(&w) = midw.get(j) {
                    if w > maxwm { maxwm = w; jmid = j; }
                }
            }
            for j in 0..=lgth2 {
                if let Some(&w) = midm.get(j) {
                    if w > maxwm { maxwm = w; jmid = j; }
                }
            }

            // Reconcile: which of {midw, midm, midn} produced max?
            let wmw = midw.get(jmid).copied().unwrap_or(f64::NEG_INFINITY);
            jumpi = imid.saturating_sub(1);
            jumpj = jmid.saturating_sub(1);
            let mut wmsel = wmw;
            if jmid > 0 {
                let nval = midn.get(jmid - 1).copied().unwrap_or(f64::NEG_INFINITY);
                if nval > wmsel {
                    jumpi = imid.saturating_sub(1);
                    jumpj = jumpbacki[jmid] as usize;
                    wmsel = nval;
                }
            }
            let mval = midm.get(jmid).copied().unwrap_or(f64::NEG_INFINITY);
            if mval > wmsel {
                jumpi = jumpbackj[jmid] as usize;
                jumpj = jmid.saturating_sub(1);
            }
            break; // C line 1782: break out of the i loop
        }

        di_signed -= 1;
    }

    // Edge cases (C lines 1721-1770): handle jmid==0 / jmid>=lgth2.
    if jmid == 0 {
        if (imid as i64) < firstmp - 1 {
            jumpi = firstmp as usize;
        }
        jmid = 1;
        jumpj = 0;
    } else if jmid >= lgth2 {
        jumpi = imid.saturating_sub(1);
        jmid = lgth2;
        jumpj = lgth2 - 1;
    } else {
        // C line 1767: `imid = jumpforwi[jumpj]; jmid = jumpforwj[jumpj]`.
        // With the backward DP fixed (`*prept` indexing) the values
        // captured at `i == imid - 1` are now byte-identical to C's
        // — verified via `rs_msalignmm_capture_top` FFI cross-validation.
        let new_imid = jumpforwi.get(jumpj).copied().unwrap_or(imid as i64);
        let new_jmid = jumpforwj.get(jumpj).copied().unwrap_or(jmid as i64);
        let mut imid_eff = imid;
        if new_imid >= 0 && new_jmid >= 0 {
            imid_eff = new_imid as usize;
            jmid = new_jmid as usize;
        }
        if imid_eff == jumpi {
            jumpi = imid_eff.saturating_sub(1);
        }
        return BackwardState {
            split_point: (jumpi, jumpj, jmid),
            imid_override: Some(imid_eff),
        };
    }

    BackwardState { split_point: (jumpi, jumpj, jmid), imid_override: None }
}

// Helper functions to compute the effective ogcp/fgcp scaled by
// penalty * nongap_freq, indexed by ABSOLUTE position in the parent
// profile. Mirrors C `MSalignmm:2298-2306`:
//   ogcp1[i] = ogcp1opt[i] * orieff1 * 0.5 * fpenalty (rough analog)
// Our `Profile.ogcp` stores the raw opening_count; the final value is
// `0.5 * (1 - opening_count) * penalty * nongap_freq`. We reuse the
// existing scaling that `profile_align_imp_with_boundary` does
// internally — and surface it here as well.

fn effective_ogcp1(p: &Profile, gap: &GapModel) -> Vec<f64> {
    let penalty = gap.open;
    let mut v: Vec<f64> = (0..p.length).map(|i| {
        0.5 * (1.0 - p.ogcp[i]) * penalty * p.nongap_freq[i]
    }).collect();
    v.push(0.0); // calloc padding
    v
}

fn effective_fgcp1(p: &Profile, gap: &GapModel) -> Vec<f64> {
    let penalty = gap.open;
    let mut v: Vec<f64> = (0..p.length).map(|i| {
        0.5 * (1.0 - p.fgcp[i]) * penalty * p.nongap_freq[i]
    }).collect();
    v.push(0.0);
    v
}

fn effective_ogcp2(p: &Profile, gap: &GapModel) -> Vec<f64> {
    effective_ogcp1(p, gap)
}

fn effective_fgcp2(p: &Profile, gap: &GapModel) -> Vec<f64> {
    effective_fgcp1(p, gap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Profile;
    use crate::dp::GapModel;

    fn simple_setup() -> (Vec<Vec<f64>>, [u8; 256]) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 { mtx[i][i] = 100.0; }
        let mut map = [4u8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map)
    }

    /// On a SMALL input (< DPTANNI), msalignmm must produce output
    /// byte-identical to `profile_align_imp_with_boundary` — because
    /// the base case just delegates to it.
    #[test]
    fn base_case_matches_full_dp() {
        let (mtx, map) = simple_setup();
        let s1: &[u8] = b"ACGTACGTAC";
        let s2: &[u8] = b"ACGTACGTAC";
        let p1 = Profile::from_aligned(&[s1], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s2], &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);

        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, true, true, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, true, true);

        assert!((aln_full.score - aln_ms.score).abs() < 1e-9,
            "scores differ: full={} ms={}", aln_full.score, aln_ms.score);
        assert_eq!(aln_full.operations, aln_ms.operations,
            "trace differs for short input");
    }

    /// Two 360-char identical-prefix sequences (length similar to
    /// the rhodopsin sample). Should align diagonally end-to-end
    /// with `head_gap=false, tail_gap=false`. If msalignmm gives a
    /// shorter trace, the recursion is broken at this length.
    #[test]
    fn freegap_360char_diagonal() {
        let (mtx, map) = simple_setup();
        let s: Vec<u8> = (0..360).map(|i| b"ACGT"[(i * 3 + 1) % 4]).collect();
        let p1 = Profile::from_aligned(&[s.as_slice()], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s.as_slice()], &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);
        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, false, false, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, false, false);
        assert_eq!(aln_full.operations.len(), aln_ms.operations.len(),
            "360-char free-gap: trace length differs: full={} ms={}",
            aln_full.operations.len(), aln_ms.operations.len());
        assert!((aln_full.score - aln_ms.score).abs() < 1e-6);
    }

    /// Asymmetric multi-seq Hirschberg (100 chars × 105-char 2-seq
    /// profile). Mirrors a typical progressive merge: a single new
    /// sequence joined to a small existing group.
    #[test]
    fn freegap_100_vs_105_multiseq() {
        let (mtx, map) = simple_setup();
        let s1: Vec<u8> = (0..100).map(|i| b"ACGT"[(i * 3 + 1) % 4]).collect();
        let s2_g1: Vec<u8> = (0..105).map(|i| b"ACGT"[(i * 7 + 5) % 4]).collect();
        let s2_g2: Vec<u8> = (0..105).map(|i| b"ACGT"[(i * 11 + 3) % 4]).collect();
        let p1 = Profile::from_aligned(&[s1.as_slice()], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s2_g1.as_slice(), s2_g2.as_slice()], &[0.5, 0.5], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);
        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, true, true, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, true, true);
        assert_eq!(aln_full.operations.len(), aln_ms.operations.len(),
            "multi-seq asymmetric: trace length differs");
        assert!((aln_full.score - aln_ms.score).abs() < 1e-6);
    }

    /// MULTI-sequence profile case (what the progressive merge
    /// actually feeds msalignmm). Verifies msalignmm handles
    /// profiles built from 2+ aligned sequences (with internal
    /// gap columns and non-trivial nongap_freq distributions).
    #[test]
    fn multiseq_profile_matches_full_dp() {
        let (mtx, map) = simple_setup();
        // Pre-aligned group of 3 sequences with internal gaps.
        let group1: Vec<&[u8]> = vec![
            b"AC-GTACGTAC-GTAC",
            b"ACGGTAC-TACGGT-C",
            b"AC-GTACGTACGGTAC",
        ];
        let group2: Vec<&[u8]> = vec![
            b"ACGTACGTAC-GTAC",
            b"ACGGTACGT-CGTAC",
        ];
        let p1 = Profile::from_aligned(&group1, &[1.0/3.0; 3], &map, 5);
        let p2 = Profile::from_aligned(&group2, &[0.5, 0.5], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);
        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, false, false, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, false, false);
        assert!((aln_full.score - aln_ms.score).abs() < 1e-6,
            "multi-seq profile: scores differ: full={} ms={}",
            aln_full.score, aln_ms.score);
        assert_eq!(aln_full.operations, aln_ms.operations,
            "multi-seq profile: trace differs");
    }

    /// `head_gap=false, tail_gap=false` case (FFT-NS-2 / NW-NS-2 path
    /// in the engine). Verifies msalignmm matches the full DP for
    /// the free-terminal-gap settings the progressive merge uses.
    #[test]
    fn freegap_matches_full_dp() {
        let (mtx, map) = simple_setup();
        let s: Vec<u8> = (0..200).map(|i| b"ACGT"[(i * 3 + 1) % 4]).collect();
        let s2: Vec<u8> = (0..200).map(|i| b"ACGT"[(i * 7 + 2) % 4]).collect();
        let p1 = Profile::from_aligned(&[s.as_slice()], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s2.as_slice()], &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);
        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, false, false, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, false, false);
        assert!((aln_full.score - aln_ms.score).abs() < 1e-6,
            "head_gap=false: scores differ: full={} ms={}", aln_full.score, aln_ms.score);
        assert_eq!(aln_full.operations.len(), aln_ms.operations.len(),
            "head_gap=false: trace length differs: full={} ms={}",
            aln_full.operations.len(), aln_ms.operations.len());
    }

    /// Recursive case with a non-trivial gap-required alignment.
    /// We embed a UNIQUE motif on either side of the gap region so
    /// the optimal alignment is uniquely determined (no tie-breaks).
    /// If the Hirschberg split logic is wrong, the score will drop.
    #[test]
    fn recursive_case_with_gap_matches_full_dp() {
        // Use the full BLOSUM-like 5-letter alphabet, with distinct
        // motifs anchoring the gap region.
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 { mtx[i][i] = 100.0; }
        let mut map = [4u8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;

        // Left half: random-ish, right half: distinct motif — gives a
        // unique optimal alignment when 20-residue insert is added.
        let left: Vec<u8> = (0..60).map(|i| b"ACGT"[(i * 3 + 1) % 4]).collect();
        let right: Vec<u8> = (0..60).map(|i| b"TGCA"[(i * 7 + 5) % 4]).collect();
        let insert: Vec<u8> = (0..20).map(|i| b"AAGG"[(i + 2) % 4]).collect();

        let s_short: Vec<u8> = left.iter().chain(right.iter()).copied().collect();
        let s_long: Vec<u8> = left.iter()
            .chain(insert.iter())
            .chain(right.iter())
            .copied().collect();
        let p1 = Profile::from_aligned(&[s_short.as_slice()], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s_long.as_slice()], &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);

        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, true, true, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, true, true);

        assert!((aln_full.score - aln_ms.score).abs() < 1e-6,
            "scores differ: full={} ms={}",
            aln_full.score, aln_ms.score);
        // Trace must match too — input has a unique optimal alignment
        // (distinct anchor motifs on either side of the gap).
        assert_eq!(aln_full.operations, aln_ms.operations,
            "trace differs for gap-required input with unique optimum");
    }

    /// Recursive case: an input long enough to trigger Hirschberg
    /// (lgth1 >= DPTANNI=100 AND lgth2 >= DPTANNI). The trace must
    /// still match the full DP, because Hirschberg gives the same
    /// optimal alignment.
    #[test]
    fn recursive_case_matches_full_dp() {
        let (mtx, map) = simple_setup();
        // Identical 200-nt sequences — the optimal alignment is
        // diagonal everywhere; Hirschberg must reproduce it exactly.
        let s: Vec<u8> = (0..200).map(|i| b"ACGT"[(i % 4) as usize]).collect();
        let p1 = Profile::from_aligned(&[s.as_slice()], &[1.0], &map, 5);
        let p2 = Profile::from_aligned(&[s.as_slice()], &[1.0], &map, 5);
        let gap = GapModel::new(-200.0, -10.0);

        let aln_full = profile_align_imp_with_boundary(
            &p1, &p2, &mtx, &gap, true, true, None, false,
            BoundaryFreqs::default(),
        );
        let aln_ms = msalignmm(&p1, &p2, &mtx, &gap, true, true);

        assert!((aln_full.score - aln_ms.score).abs() < 1.0,
            "scores differ: full={} ms={}", aln_full.score, aln_ms.score);
        // The trace MUST be byte-identical for identical inputs.
        assert_eq!(aln_full.operations.len(), aln_ms.operations.len(),
            "trace length differs: full={} ms={}",
            aln_full.operations.len(), aln_ms.operations.len());
        assert_eq!(aln_full.operations, aln_ms.operations,
            "trace differs for identical 200-nt input");
    }
}
