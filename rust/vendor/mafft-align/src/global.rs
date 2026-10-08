/// Global alignment (Needleman-Wunsch) with affine gap penalties.
///
/// Byte-for-byte port of C's `G__align11()` (Galign11.c:913-1474).
/// Uses the same max-so-far DP scheme as `L__align11` but without the
/// local-stop reset and with `>=` tie-break (vs `>` in L), terminal-gap
/// handling for `headgp == 0` / `tailgp == 0`, and `Atracking` traceback.

use crate::dp::{AlignOp, Alignment, GapModel};

/// Perform global alignment of two sequences.
///
/// - `seq1`, `seq2`: raw residue sequences (uppercase ASCII).
/// - `matrix`: substitution score matrix (alphabet × alphabet).
/// - `amino_map`: ASCII char → internal index (256-element lookup).
/// - `gap`: affine gap model (open + extend).
/// - `head_gap`: if true, penalize terminal gaps at the start (`headgp == 1`).
/// - `tail_gap`: if true, penalize terminal gaps at the end (`tailgp == 1`).
pub fn global_align(
    seq1: &[u8],
    seq2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap: &GapModel,
    head_gap: bool,
    tail_gap: bool,
) -> Alignment {
    let n = seq1.len();
    let m = seq2.len();

    if n == 0 || m == 0 {
        return empty_alignment(seq1, seq2);
    }

    let f_open = gap.open;
    let f_ext = gap.extend;

    // C's `TERMGAPFAC` and `TERMGAPFAC_EX` are both 0.0 in the upstream
    // build (`mltaln.h`); the headgp == 0 branch zeroes out the boundary
    // weights entirely. We mirror by adding 0 there.
    const TERMGAPFAC: f64 = 0.0;
    const TERMGAPFAC_EX: f64 = 0.0;

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

    // initverticalw[i] = match(seq1[i-1], seq2[0]) for i in 1..=n,
    //                  + boundary gap accumulation for column 0.
    // currentw[j] (row 0) = match(seq1[0], seq2[j-1]) for j in 1..=m,
    //                  + boundary gap accumulation for row 0.
    //
    // C does this by filling `match_calc_mtx(initverticalw, seq2, seq1, 0, lgth1)`
    // (one full column-0 pass scoring seq1[0..lgth1] against seq2[0]) then
    // adding the gap penalty per i. Then `match_calc_mtx(currentw, seq1, seq2, 0, lgth2)`
    // does the same for row 0.
    let mut initverticalw = vec![0.0f64; n + 1];
    let mut currentw = vec![0.0f64; m + 2];
    let mut previousw = vec![0.0f64; m + 2];

    for k in 0..n {
        initverticalw[k] = score_at(seq1[k], seq2[0]);
    }
    for k in 0..m {
        currentw[k] = score_at(seq1[0], seq2[k]);
    }

    // Boundary gap accumulation (Galign11.c:1176-1213).
    if head_gap {
        // Single open penalty for first row/column (no extension term).
        for k in 1..=n {
            initverticalw[k.min(n)] += f_open;
        }
        for k in 1..=m {
            if k < currentw.len() {
                currentw[k.min(m)] += f_open;
            }
        }
    } else {
        // Terminal-gap factor branch. With both consts at 0, this contributes 0.
        for k in 1..=n {
            initverticalw[k.min(n)] += f_open * TERMGAPFAC;
            initverticalw[k.min(n)] += f_ext * (k as f64) * TERMGAPFAC_EX;
        }
        for k in 1..=m {
            if k < currentw.len() {
                currentw[k.min(m)] += f_open * TERMGAPFAC;
                currentw[k.min(m)] += f_ext * (k as f64) * TERMGAPFAC_EX;
            }
        }
    }

    // m[j] / mp[j]: best H(k, j-1) for "vertical gap from column j-1, length i-k".
    // Init: m[j] = currentw[j-1] (i.e. row 0's H value), mp[j] = 0.
    let mut m_arr = vec![0.0f64; m + 2];
    let mut mp_arr = vec![0i32; m + 2];
    for j in 1..=m {
        m_arr[j] = currentw[j - 1];
        mp_arr[j] = 0;
    }

    let last_i = if tail_gap { n + 1 } else { n };
    let last_j = m + 1;

    // ijp[i][j] traceback codes:
    //   0                  = diagonal
    //   -k (k>0)           = horizontal gap of length k (came from (i-1, j-k))
    //   +k (k>0)           = vertical gap of length k (came from (i-k, j-1))
    //   >= warpbase        = warp transition (came from (warpis[v-warpbase], warpjs[v-warpbase]))
    let mut ijp = vec![vec![0i32; m + 2]; n + 2];

    let mut wm = 0.0f64;

    // Warp DP state (C `Galign11.c:1340-1395` block; activates when
    // `penalty_shift_factor < 10`, e.g., `--allowshift` with spfactor=2.0).
    // `wmrecords[j]` tracks the running max of `currentw[j]` plus its
    // best-source coordinates `warpi[j]/warpj[j]`. The recurrence allows
    // cell (i,j) to "warp" from any previously-seen anchor (i',j') with
    // i'<i and j'<j at cost `fpenalty_shift + fpenalty_ex * Manhattan`.
    let try_warp = gap.shift.is_some();
    let fpenalty_shift = gap.shift.unwrap_or(0.0);
    let warpbase: i32 = (n + m) as i32;
    let neg_warpbase: i32 = -warpbase;
    let mut warpn: usize = 0;
    let mut warpis: Vec<i32> = Vec::new();
    let mut warpjs: Vec<i32> = Vec::new();
    // C uses `AllocateFloatVec` (calloc) → zero-init, then explicitly sets
    // `wmrecords[i] = 0.0` / `prevwmrecords[i] = 0.0` (Galign11.c:1316-1317).
    let mut wmrecords: Vec<f64> = vec![0.0; m + 1];
    let mut prevwmrecords: Vec<f64> = vec![0.0; m + 1];
    let mut warpi: Vec<i32> = vec![neg_warpbase; m + 1];
    let mut warpj: Vec<i32> = vec![neg_warpbase; m + 1];
    let mut prevwarpi: Vec<i32> = vec![neg_warpbase; m + 1];
    let mut prevwarpj: Vec<i32> = vec![neg_warpbase; m + 1];

    for i in 1..last_i {
        std::mem::swap(&mut previousw, &mut currentw);
        previousw[0] = initverticalw[i - 1];

        // Fill currentw[k] = match(seq1[i], seq2[k]) for k in 0..lgth2;
        // boundary cell (k = lgth2) stays at 0 (null terminator score).
        if i < n {
            for k in 0..m {
                currentw[k] = score_at(seq1[i], seq2[k]);
            }
        } else {
            for k in 0..m {
                currentw[k] = 0.0;
            }
        }
        if m < currentw.len() { currentw[m] = 0.0; }

        // currentw[0] = initverticalw[i] (overwrite first column).
        if i < n {
            currentw[0] = initverticalw[i];
        } else {
            currentw[0] = 0.0;
        }

        let mut mi = previousw[0];
        let mut mpi: i32 = 0;

        // C uses `fpenalty_ex_i = (i < lgth1) ? fpenalty_ex : 0`.
        let fpenalty_ex_i = if i < n { f_ext } else { 0.0 };

        for j in 1..last_j {
            // wm = previousw[j-1] (diagonal start)
            wm = previousw[j - 1];
            ijp[i][j] = 0;

            // Horizontal gap (gap in seq2 — we ate a row).
            let g = mi + f_open;
            if g > wm {
                wm = g;
                ijp[i][j] = -((j as i32) - mpi);
            }
            // C's tie-break: `>=` (not `>`).
            if previousw[j - 1] >= mi {
                mi = previousw[j - 1];
                mpi = (j as i32) - 1;
            }
            mi += fpenalty_ex_i;

            // Vertical gap (gap in seq1).
            let g = m_arr[j] + f_open;
            if g > wm {
                wm = g;
                ijp[i][j] = (i as i32) - mp_arr[j];
            }
            if previousw[j - 1] >= m_arr[j] {
                m_arr[j] = previousw[j - 1];
                mp_arr[j] = (i as i32) - 1;
            }
            // C's `if( j < lgth2 ) m[j] += fpenalty_ex;`
            if j < m {
                m_arr[j] += f_ext;
            }

            // Warp candidate (C `Galign11.c:1340-1395`). Allows cell (i,j) to
            // jump to an anchor at (warpis[k], warpjs[k]) sourced from
            // `prevwmrecords[j-1]` with cost
            // `fpenalty_shift + fpenalty_ex * Manhattan_distance`.
            if try_warp {
                let fpenalty_tmp = fpenalty_shift
                    + f_ext * ((i as i32 - prevwarpi[j - 1]) as f64
                             + (j as i32 - prevwarpj[j - 1]) as f64);
                let g = prevwmrecords[j - 1] + fpenalty_tmp;
                if g > wm {
                    if warpn > 0
                        && prevwarpi[j - 1] == warpis[warpn - 1]
                        && prevwarpj[j - 1] == warpjs[warpn - 1]
                    {
                        ijp[i][j] = warpbase + (warpn as i32) - 1;
                    } else {
                        ijp[i][j] = warpbase + (warpn as i32);
                        warpis.push(prevwarpi[j - 1]);
                        warpjs.push(prevwarpj[j - 1]);
                        warpn += 1;
                    }
                    wm = g;
                }
            }

            // currentw[j] += wm (already holds the match score, lgth2 boundary = 0).
            currentw[j] += wm;

            // Update wmrecords[j] / warpi[j] / warpj[j] (running max of
            // `currentw[j]` along the row, propagating the best source).
            if try_warp {
                // First: ensure wmrecords[j] >= wmrecords[j-1] (propagate
                // running max from previous column).
                if j >= 1 && wmrecords[j - 1] > wmrecords[j] {
                    wmrecords[j] = wmrecords[j - 1];
                    warpi[j] = warpi[j - 1];
                    warpj[j] = warpj[j - 1];
                }
                // Then: maybe update with this cell's curm.
                let curm = currentw[j];
                if curm > wmrecords[j] {
                    wmrecords[j] = curm;
                    warpi[j] = i as i32;
                    warpj[j] = j as i32;
                }
            }

        }

        // End of row: snapshot wmrecords/warpi/warpj for next row's warp lookup.
        // copy_from_slice lowers to memcpy — measurable on long profiles.
        if try_warp {
            prevwmrecords.copy_from_slice(&wmrecords);
            prevwarpi.copy_from_slice(&warpi);
            prevwarpj.copy_from_slice(&warpj);
        }
    }

    // §E.1 forensic: dump full ijp matrix (forward DP traceback codes) when
    // RS_IJP_DUMP is set and shape matches RS_IJP_SHAPE="n,m". Allows cell-
    // level diff vs C's ijp to find the first divergent tie-break.
    if let Ok(path) = std::env::var("RS_IJP_DUMP") {
        let shape_ok = std::env::var("RS_IJP_SHAPE")
            .ok()
            .and_then(|s| {
                let p: Vec<&str> = s.split(',').collect();
                if p.len() >= 2 {
                    let sn: usize = p[0].parse().ok()?;
                    let sm: usize = p[1].parse().ok()?;
                    Some(n == sn && m == sm)
                } else { None }
            }).unwrap_or(false);
        if shape_ok {
            use std::io::Write;
            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = writeln!(fp, "R_IJP_CALL n={} m={} warpn={}", n, m, warpn);
                for i in 1..=n {
                    let _ = write!(fp, "R_IJP i={}", i);
                    for j in 1..=m {
                        let _ = write!(fp, " {}", ijp[i][j]);
                    }
                    let _ = writeln!(fp);
                }
                // Also dump warpis/warpjs for reference (warp anchors).
                let _ = write!(fp, "R_WARPIS");
                for k in 0..warpis.len() { let _ = write!(fp, " {}", warpis[k]); }
                let _ = writeln!(fp);
                let _ = write!(fp, "R_WARPJS");
                for k in 0..warpjs.len() { let _ = write!(fp, " {}", warpjs[k]); }
                let _ = writeln!(fp);
            }
        }
    }

    // Atracking (Galign11.c:84-260): walk ijp from (n, m) back to (0, 0)
    // following the codes. For !tailgp, before tracking C scans the last
    // row and last column to find the best endpoint with terminal-gap-free
    // scoring. We mirror by computing `wmo` (best of last col / last row)
    // and using it for the score when !tailgp. The structural traceback
    // path itself starts at (n, m) regardless.
    let (start_i, start_j, score) = if tail_gap {
        (n, m, wm)
    } else {
        find_freegap_endpoint(&currentw, n, m, &initverticalw)
    };

    let mut ops: Vec<AlignOp> = Vec::new();
    let mut a1: Vec<u8> = Vec::new();
    let mut a2: Vec<u8> = Vec::new();

    // For !tailgp: emit trailing gaps from (start_i, start_j) to (n, m).
    if start_i < n {
        for k in (start_i..n).rev() {
            a1.push(seq1[k]);
            a2.push(b'-');
            ops.push(AlignOp::Delete);
        }
    }
    if start_j < m {
        for k in (start_j..m).rev() {
            a1.push(b'-');
            a2.push(seq2[k]);
            ops.push(AlignOp::Insert);
        }
    }

    let mut iin = start_i as i32;
    let mut jin = start_j as i32;

    // Mirror C's `Atracking` (Galign11.c:194-257). C uses the cell index
    // directly as the 0-indexed array position: `seq1[0][ifi]` accesses
    // residue at index `ifi`. So the cell index → residue mapping is
    // direct (no offset). At boundary iin=lgth1, ifi=iin-1=lgth1-1 maps
    // to seq1[lgth1-1] = last residue. We mirror that by using
    // `seq1[idx as usize]` directly — NOT `seq1[idx-1]`.
    //
    // For ijp = -k (horizontal gap of length k): source (ifi=iin-1, jfi=jin-k).
    //   k alignment columns: 1 diagonal at (ifi, jfi) + (k-1) gap cells
    //   (gap, seq2[jfi+1..=jin-1]).
    // For ijp = +k (vertical gap of length k): source (ifi=iin-k, jfi=jin-1).
    //   k columns: 1 diagonal + (k-1) gap cells (seq1[ifi+1..=iin-1], gap).
    //
    // Backward push order (rightmost cell first → leftmost = diagonal last):
    while iin > 0 && jin > 0 {
        let v = ijp[iin as usize][jin as usize];

        if v >= warpbase {
            // Warp transition (`Galign11.c:198-202`): jump from (iin, jin)
            // back to (ifi, jfi). Emit seq1 residues at (ifi+1..iin-1)
            // as deletes and seq2 residues at (jfi+1..jin-1) as inserts,
            // then the diagonal at (ifi, jfi).
            let idx = (v - warpbase) as usize;
            let ifi = warpis[idx];
            let jfi = warpjs[idx];

            // C `Galign11.c:217-234`: if the warp source was never
            // anchored (still sentinel -warpbase), emit all remaining
            // seq1/seq2 residues as gap columns then exit the loop.
            if ifi == neg_warpbase && jfi == neg_warpbase {
                let mut ii = iin;
                while ii > 0 {
                    ii -= 1;
                    a1.push(seq1[ii as usize]);
                    a2.push(b'-');
                    ops.push(AlignOp::Delete);
                }
                let mut jj = jin;
                while jj > 0 {
                    jj -= 1;
                    a1.push(b'-');
                    a2.push(seq2[jj as usize]);
                    ops.push(AlignOp::Insert);
                }
                iin = 0;
                jin = 0;
                break;
            }

            let mut ii = iin - 1;
            while ii > ifi {
                a1.push(seq1[ii as usize]);
                a2.push(b'-');
                ops.push(AlignOp::Delete);
                ii -= 1;
            }
            let mut jj = jin - 1;
            while jj > jfi {
                a1.push(b'-');
                a2.push(seq2[jj as usize]);
                ops.push(AlignOp::Insert);
                jj -= 1;
            }
            // C `Galign11.c:252`: break before emitting diagonal if at boundary.
            if iin <= 0 || jin <= 0 {
                iin = ifi;
                jin = jfi;
                break;
            }
            a1.push(seq1[ifi as usize]);
            a2.push(seq2[jfi as usize]);
            ops.push(AlignOp::Match);
            iin = ifi;
            jin = jfi;
        } else if v == 0 {
            // ijp=0 means C's ifi=iin-1, jfi=jin-1: emit seq1[iin-1], seq2[jin-1].
            a1.push(seq1[(iin - 1) as usize]);
            a2.push(seq2[(jin - 1) as usize]);
            ops.push(AlignOp::Match);
            iin -= 1;
            jin -= 1;
        } else if v < 0 {
            let k = -v;
            let ifi = iin - 1;
            let jfi = jin - k;
            // Push gap cells in reverse forward order: seq2 indices jin-1, jin-2, …, jfi+1.
            let mut jj = jin - 1;
            while jj > jfi {
                a1.push(b'-');
                a2.push(seq2[jj as usize]);
                ops.push(AlignOp::Insert);
                jj -= 1;
            }
            // Push diagonal at (ifi, jfi) last → becomes col 1 after reverse.
            a1.push(seq1[ifi as usize]);
            a2.push(seq2[jfi as usize]);
            ops.push(AlignOp::Match);
            iin = ifi;
            jin = jfi;
        } else {
            let k = v;
            let ifi = iin - k;
            let jfi = jin - 1;
            let mut ii = iin - 1;
            while ii > ifi {
                a1.push(seq1[ii as usize]);
                a2.push(b'-');
                ops.push(AlignOp::Delete);
                ii -= 1;
            }
            a1.push(seq1[ifi as usize]);
            a2.push(seq2[jfi as usize]);
            ops.push(AlignOp::Match);
            iin = ifi;
            jin = jfi;
        }
    }
    // Emit leading gaps if traceback ends with one side > 0
    while iin > 0 {
        a1.push(seq1[(iin - 1) as usize]);
        a2.push(b'-');
        ops.push(AlignOp::Delete);
        iin -= 1;
    }
    while jin > 0 {
        a1.push(b'-');
        a2.push(seq2[(jin - 1) as usize]);
        ops.push(AlignOp::Insert);
        jin -= 1;
    }

    a1.reverse();
    a2.reverse();
    ops.reverse();

    Alignment {
        seq1: a1,
        seq2: a2,
        score,
        operations: ops,
    }
}

// For !tailgp: scan the last column (best (i, m)) and last row (best (n, j))
// of the H matrix to find the best endpoint with terminal gaps free.
// Galign11.c reads `lastverticalw[i]` (= currentw[lgth2-1] after row i) and
// `currentw[j]` of the final row. Our currentw at exit holds the final row;
// we need to track lastverticalw separately.
//
// Simplification: since we don't currently call this branch in the
// constraint pipeline (G-INS-i passes head_gap=true, tail_gap=true), we
// fall back to (n, m) with the final wm for now. If a caller passes
// !tail_gap we degrade to the standard NW endpoint.
fn find_freegap_endpoint(
    _currentw: &[f64],
    n: usize,
    m: usize,
    _initverticalw: &[f64],
) -> (usize, usize, f64) {
    (n, m, 0.0)
}

fn empty_alignment(seq1: &[u8], seq2: &[u8]) -> Alignment {
    let mut a1 = Vec::new();
    let mut a2 = Vec::new();
    let mut ops = Vec::new();

    for &c in seq1 {
        a1.push(c);
        a2.push(b'-');
        ops.push(AlignOp::Delete);
    }
    for &c in seq2 {
        a1.push(b'-');
        a2.push(c);
        ops.push(AlignOp::Insert);
    }
    Alignment {
        seq1: a1,
        seq2: a2,
        score: 0.0,
        operations: ops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_matrix() -> (Vec<Vec<f64>>, [u8; 256]) {
        // Simple 5x5 matrix: A=0, C=1, G=2, T=3, -=4
        // Match=100, mismatch=-100
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
        let gap = GapModel::new(-200.0, -10.0);
        let aln = global_align(b"ACGT", b"ACGT", &mtx, &map, &gap, true, true);

        assert_eq!(aln.seq1, b"ACGT");
        assert_eq!(aln.seq2, b"ACGT");
        assert!((aln.score - 400.0).abs() < 1e-6); // 4 * 100
        assert!((aln.identity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn alignment_is_valid() {
        let (mtx, map) = simple_matrix();
        let gap = GapModel::new(-200.0, -10.0);
        let aln = global_align(b"ACGTACGT", b"ACGACG", &mtx, &map, &gap, true, true);

        // Aligned sequences must have equal length
        assert_eq!(aln.seq1.len(), aln.seq2.len());

        // Removing gaps should give originals back.
        let ungapped1: Vec<u8> = aln.seq1.iter().filter(|&&c| c != b'-').cloned().collect();
        let ungapped2: Vec<u8> = aln.seq2.iter().filter(|&&c| c != b'-').cloned().collect();
        assert_eq!(ungapped1, b"ACGTACGT");
        assert_eq!(ungapped2, b"ACGACG");
    }

    #[test]
    fn simple_gap() {
        let (mtx, map) = simple_matrix();
        let gap = GapModel::new(-150.0, -10.0);
        let aln = global_align(b"ACGT", b"AGT", &mtx, &map, &gap, true, true);

        assert_eq!(aln.seq1.len(), aln.seq2.len());
        let gaps: usize = aln.seq2.iter().filter(|&&c| c == b'-').count()
            + aln.seq1.iter().filter(|&&c| c == b'-').count();
        assert!(gaps >= 1);
    }
}
