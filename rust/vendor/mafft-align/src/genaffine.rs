/// Generalized affine gap local alignment (for E-INS-i).
///
/// Byte-for-byte port of C's `genL__align11()` (genalign11.c:113-660).
/// Same max-so-far DP scheme as `L__align11` plus a "skip" gap state with
/// its own opening penalty (`penalty_OP`) and zero extension cost. The
/// skip mode handles long unaligned regions cheaply where a regular
/// affine gap would accumulate prohibitive extension penalties.
///
/// State per row:
///   `mi`, `mpi`     — running max for vertical gap from row i with extension.
///   `m_arr[j]`, `mp_arr[j]` — running max for horizontal gap to col j with extension.
///   `tbk`, `tbki`, `tbkj` — running max for "skip" path within current row.
///   `Mi`, `Mpi`     — running max of `previousw[j-1]` along row.
///   `largeM[j]`, `Mp[j]` — running max of `previousw[j-1]` along col.
///
/// Traceback uses two arrays `ijpi[i][j]` / `ijpj[i][j]` storing absolute
/// source coordinates (not relative offsets), so a single jump can change
/// both i and j arbitrarily.

use crate::dp::{AlignOp, Alignment, GapModel};
use crate::local::LocalAlignment;

/// Extended gap model with a separate "generalized" opening penalty
/// (corresponds to C's `penalty_OP`).
#[derive(Debug, Clone)]
pub struct GenAffineGapModel {
    /// Standard affine gap model (`penalty`, `penalty_ex`).
    pub affine: GapModel,
    /// Generalized "skip" opening penalty (C's `penalty_OP`). No extension.
    pub open_generalized: f64,
}

impl Default for GenAffineGapModel {
    fn default() -> Self {
        Self {
            affine: GapModel::default(),
            open_generalized: -918.0,
        }
    }
}

/// Perform generalized-affine local alignment.
///
/// - `seq1`, `seq2`: raw residue sequences (uppercase ASCII).
/// - `matrix`: substitution score matrix (alphabet × alphabet).
/// - `amino_map`: ASCII char → internal index.
/// - `gap_model`: affine penalties + skip-open penalty.
/// - `score_offset`: subtract this * 600 from the local-stop threshold,
///   matching C's `localthr = -offset` where `offset = (int)(scale*poffset+0.5)`.
pub fn genaffine_local_align(
    seq1: &[u8],
    seq2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap_model: &GenAffineGapModel,
    score_offset: f64,
) -> LocalAlignment {
    let n = seq1.len();
    let m = seq2.len();

    if n == 0 || m == 0 {
        return LocalAlignment {
            alignment: Alignment {
                seq1: Vec::new(),
                seq2: Vec::new(),
                score: 0.0,
                operations: Vec::new(),
            },
            offset1: 0,
            offset2: 0,
        };
    }

    let f_open = gap_model.affine.open;
    let f_ext = gap_model.affine.extend;
    let f_op = gap_model.open_generalized;
    // C's `localthr = -foffset = -score_offset*600` (genalign11.c:161-162).
    let localthr = -score_offset * 600.0;

    let n_alpha = matrix.len();
    let score_at = |c1: u8, c2: u8| -> f64 {
        let i = amino_map[c1 as usize] as usize;
        let j = amino_map[c2 as usize] as usize;
        if i < n_alpha && j < n_alpha {
            matrix[i][j]
        } else {
            0.0
        }
    };

    // initverticalw[k] = match(seq1[k], seq2[0]) for k in 0..n
    let mut initverticalw = vec![0.0f64; n + 1];
    for k in 0..n {
        initverticalw[k] = score_at(seq1[k], seq2[0]);
    }
    let mut currentw = vec![0.0f64; m + 1];
    let mut previousw = vec![0.0f64; m + 1];
    for k in 0..m {
        currentw[k] = score_at(seq1[0], seq2[k]);
    }

    // m[j], mp[j], largeM[j], Mp[j] init: m[j] = currentw[j-1], mp = 0.
    let mut m_arr = vec![0.0f64; m + 1];
    let mut mp_arr = vec![0i32; m + 1];
    let mut large_m = vec![0.0f64; m + 1];
    let mut large_mp = vec![0i32; m + 1];
    for j in 1..=m {
        m_arr[j] = currentw[j - 1];
        mp_arr[j] = 0;
        large_m[j] = currentw[j - 1];
        large_mp[j] = 0;
    }

    // Special markers per genalign11.c:357 — `localstop = lgth1+lgth2+1`.
    // We use i32::MIN for clarity; the only thing the traceback checks is
    // `ijpi[i][j] == localstop`.
    const LOCALSTOP: i32 = i32::MIN;
    let mut ijpi = vec![vec![0i32; m + 1]; n + 1];
    let mut ijpj = vec![vec![0i32; m + 1]; n + 1];
    for i in 0..=n { ijpi[i][0] = LOCALSTOP; ijpj[i][0] = LOCALSTOP; }
    for j in 0..=m { ijpi[0][j] = LOCALSTOP; ijpj[0][j] = LOCALSTOP; }

    let mut maxwm = f64::NEG_INFINITY;
    let mut endali = 0i32;
    let mut endalj = 0i32;

    for i in 1..=n {
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[0] = initverticalw[i - 1];

        // Fill currentw with match scores for row i. Boundary i=n stays 0.
        if i < n {
            for k in 0..m {
                currentw[k] = score_at(seq1[i], seq2[k]);
            }
        } else {
            for k in 0..m { currentw[k] = 0.0; }
        }

        currentw[0] = if i < n { initverticalw[i] } else { 0.0 };

        let mut mi = previousw[0];
        let mut mpi: i32 = 0;
        let mut large_mi = previousw[0];
        let mut large_mpi: i32 = 0;
        let mut tbk: f64 = -1e9;
        let mut tbki: i32 = 0;
        let mut tbkj: i32 = 0;

        for j in 1..=m {
            let mut wm = previousw[j - 1];
            ijpi[i][j] = (i as i32) - 1;
            ijpj[i][j] = (j as i32) - 1;

            // Vertical gap: open + extend (mi tracks running max in row i).
            let g = mi + f_open;
            if g > wm {
                wm = g;
                ijpj[i][j] = mpi;
                // ijpi[i][j] stays at i-1
            }
            if previousw[j - 1] > mi {
                mi = previousw[j - 1];
                mpi = (j as i32) - 1;
            }
            mi += f_ext;

            // Horizontal gap: open + extend (m[j] tracks running max in col j).
            let g = m_arr[j] + f_open;
            if g > wm {
                wm = g;
                ijpi[i][j] = mp_arr[j];
                ijpj[i][j] = (j as i32) - 1;
            }
            if previousw[j - 1] > m_arr[j] {
                m_arr[j] = previousw[j - 1];
                mp_arr[j] = (i as i32) - 1;
            }
            m_arr[j] += f_ext;

            // Skip / generalized gap: penalty_OP only, no extension.
            let g = tbk + f_op;
            if g > wm {
                wm = g;
                ijpi[i][j] = tbki;
                ijpj[i][j] = tbkj;
            }
            // Update tbk from row-running-max Mi.
            if large_mi > tbk {
                tbk = large_mi;
                tbki = (i as i32) - 1;
                tbkj = large_mpi;
            }
            // Update tbk from col-running-max largeM[j].
            if large_m[j] > tbk {
                tbk = large_m[j];
                tbki = large_mp[j];
                tbkj = (j as i32) - 1;
            }
            // Update column running-max from previousw[j-1].
            if previousw[j - 1] > large_m[j] {
                large_m[j] = previousw[j - 1];
                large_mp[j] = (i as i32) - 1;
            }
            // Update row running-max.
            if previousw[j - 1] > large_mi {
                large_mi = previousw[j - 1];
                large_mpi = (j as i32) - 1;
            }

            if maxwm < wm {
                maxwm = wm;
                endali = i as i32;
                endalj = j as i32;
            }

            // localthr clipping → store localstop marker, reset wm.
            if wm < localthr {
                ijpi[i][j] = LOCALSTOP;
                wm = localthr;
            }

            currentw[j] += wm;
        }
    }

    if endali == 0 || endalj == 0 || maxwm <= 0.0 {
        return LocalAlignment {
            alignment: Alignment {
                seq1: Vec::new(),
                seq2: Vec::new(),
                score: 0.0,
                operations: Vec::new(),
            },
            offset1: 0,
            offset2: 0,
        };
    }

    // Traceback (gentracking, genalign11.c:30-110). Walk ijpi/ijpj from
    // (endali, endalj) until a localstop or i<=0 or j<=0. Each step jumps
    // to the absolute source (ifi, jfi) and emits:
    //   (iin-ifi-1) cells (seq1[ifi+1..=iin-1], gap)  -- "vertical" leg
    //   (jin-jfi-1) cells (gap, seq2[jfi+1..=jin-1])  -- "horizontal" leg
    //   1 diagonal cell at (seq1[ifi], seq2[jfi]).
    let mut ops: Vec<AlignOp> = Vec::new();
    let mut a1: Vec<u8> = Vec::new();
    let mut a2: Vec<u8> = Vec::new();
    let mut iin = endali;
    let mut jin = endalj;
    let mut last_ifi = iin;
    let mut last_jfi = jin;

    loop {
        if iin <= 0 || jin <= 0 { break; }
        if (iin as usize) > n || (jin as usize) > m { break; }
        let ifi = ijpi[iin as usize][jin as usize];
        let jfi = ijpj[iin as usize][jin as usize];

        if ifi == LOCALSTOP || jfi == LOCALSTOP {
            last_ifi = iin;
            last_jfi = jin;
            break;
        }

        // Push gap cells (rightmost first so reverse() yields forward order:
        // diagonal → seq2-gaps → seq1-gaps … wait — actually genL gentracking
        // emits vertical-leg gaps first, then horizontal-leg gaps, then
        // diagonal. We mirror by pushing in the same backward-traceback order
        // and reversing.
        // C order at each step (writes go to lower memory offsets):
        //   vertical-leg gaps:   (seq1[ifi+l], gap) for l = (iin-ifi-1)..=1
        //   horizontal-leg gaps: (gap, seq2[jfi+l]) for l = (jin-jfi-1)..=1
        //   diagonal:            (seq1[ifi], seq2[jfi])
        // After reverse, forward order is:
        //   diagonal, horizontal-leg gaps left-to-right, vertical-leg gaps left-to-right.
        let mut l = iin - ifi;
        while l > 1 {
            l -= 1;
            let idx = (ifi + l) as usize;
            if idx < n {
                a1.push(seq1[idx]);
                a2.push(b'-');
                ops.push(AlignOp::Delete);
            }
        }
        let mut l = jin - jfi;
        while l > 1 {
            l -= 1;
            let idx = (jfi + l) as usize;
            if idx < m {
                a1.push(b'-');
                a2.push(seq2[idx]);
                ops.push(AlignOp::Insert);
            }
        }

        // Boundary: if iin or jin is past the end, skip residue emission for
        // the current cell (mirrors L__align11 boundary). Otherwise emit
        // diagonal at (ifi, jfi).
        if (ifi as usize) < n && (jfi as usize) < m && ifi >= 0 && jfi >= 0 {
            a1.push(seq1[ifi as usize]);
            a2.push(seq2[jfi as usize]);
            ops.push(AlignOp::Match);
        }
        last_ifi = ifi;
        last_jfi = jfi;

        // Check next cell for localstop.
        if ifi <= 0 || jfi <= 0 { break; }
        let next_ifi = ijpi[ifi as usize][jfi as usize];
        let next_jfi = ijpj[ifi as usize][jfi as usize];
        if next_ifi == LOCALSTOP || next_jfi == LOCALSTOP { break; }
        iin = ifi;
        jin = jfi;
    }

    a1.reverse();
    a2.reverse();
    ops.reverse();

    let offset1 = last_ifi.max(0) as usize;
    let offset2 = last_jfi.max(0) as usize;

    LocalAlignment {
        alignment: Alignment {
            seq1: a1,
            seq2: a2,
            score: maxwm,
            operations: ops,
        },
        offset1,
        offset2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_matrix() -> (Vec<Vec<f64>>, [u8; 256]) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 {
            mtx[i][i] = 100.0;
        }
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0;
        map[b'C' as usize] = 1;
        map[b'G' as usize] = 2;
        map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map)
    }

    #[test]
    fn identical_sequences() {
        let (mtx, map) = simple_matrix();
        let gap = GenAffineGapModel {
            affine: GapModel::new(-200.0, -10.0),
            open_generalized: -300.0,
        };
        let result = genaffine_local_align(b"ACGT", b"ACGT", &mtx, &map, &gap, 0.0);
        assert!((result.alignment.score - 400.0).abs() < 1e-6);
    }

    #[test]
    fn handles_long_insertion() {
        let (mtx, map) = simple_matrix();
        let gap = GenAffineGapModel {
            affine: GapModel::new(-200.0, -50.0),
            open_generalized: -150.0,
        };
        let s1 = b"ACGTACGT";
        let s2 = b"ACGTTTTTTTTTTTTTACGT";
        let result = genaffine_local_align(s1, s2, &mtx, &map, &gap, 0.0);
        assert!(result.alignment.score > 200.0);
    }

    #[test]
    fn score_non_negative() {
        let (mtx, map) = simple_matrix();
        let gap = GenAffineGapModel::default();
        let result = genaffine_local_align(b"AAAA", b"CCCC", &mtx, &map, &gap, 0.0);
        assert!(result.alignment.score >= 0.0);
    }
}
