//! Multi-distance-class match-score kernel for `--allowshift` refinement.
//!
//! Port of C MAFFT's `partA__align_variousdist` match computation
//! (`partSalignmm.c`): instead of one substitution matrix, each sequence
//! pair `(i, j)` is scored with a distance-appropriate matrix
//! `matrices[which[i][j]]`. The naive per-pair sum is O(clus1·clus2) per
//! cell, so C uses a cpmx (column-profile) decomposition:
//!
//!   match[k] = Σ_c  cpmx1s[c][i1] · Mᶜ · cpmx2s[c][k]      (match_calc_add)
//!            − Σ_c  Σ_{(i,j)∈mask[c]} Mᶜ[res_i][res_j]·eff1[i]·eff2[j]  (match_calc_del)
//!
//! The per-class profiles `cpmx1s[c]` carry weight `eff1[i]` for sequence
//! `i` iff `i` appears in *some* pair of class `c`; the cpmx product then
//! over-counts pairs `(i,j)` whose true class `which[i][j] ≠ c` but whose
//! endpoints both happen to land in class `c`. `match_calc_del` subtracts
//! exactly those spurious contributions (the `mask` lists), leaving each
//! pair counted once with its own matrix. See `tditeration.c::classifypairs`
//! for how `which` / `eff1s` / `eff2s` (hence `cpmx1s`/`cpmx2s`) are built.
//!
//! FP NOTE: the summation order must match C — per-class accumulation in
//! ascending `c`, the inner `scarr`/sparse dots with `mul_add` (C fuses to
//! FMA under `-O3`), then the `del` subtraction in `c`, `k`, `m` order.

/// Per-branch multi-matrix context, built once per refinement realign from
/// `varidist::{make_scoring_matrices, classify_pairs}` +
/// `BranchWeights::dist_from_a_branch`.
pub struct MultiMtx<'a> {
    /// `matrices[c][a][b]` — substitution matrix for distance class `c`.
    pub matrices: &'a [Vec<Vec<f64>>],
    /// Per-class column profiles of cluster 1: `cpmx1s[c][col][alpha]`
    /// (weighted by `eff1s[c]`). Same `[col][alpha]` layout as `Profile.freqs`.
    pub cpmx1s: &'a [Vec<Vec<f64>>],
    /// Per-class column profiles of cluster 2: `cpmx2s[c][col][alpha]`.
    pub cpmx2s: &'a [Vec<Vec<f64>>],
    /// Sparse representation of `cpmx1s` / `cpmx2s` — per class, per column,
    /// `Vec<(alpha_index, value)>` of the non-zero alphabet entries only.
    /// Iterating these instead of the dense `0..nalpha` skips zero-weight
    /// alphabet positions, which is a large win for single- or few-residue
    /// clusters (typical in refinement). FP order preserved because the
    /// dense iteration's zero contributions are `x.mul_add(0, acc) = acc`,
    /// so dropping them is mathematically identical.
    pub cpmx1s_sparse: &'a [Vec<Vec<(u8, f64)>>],
    pub cpmx2s_sparse: &'a [Vec<Vec<(u8, f64)>>],
    /// Spurious-pair masks per class (pairs in the class's cpmx product whose
    /// true class differs). `mask1[c]`/`mask2[c]` index cluster-1/2 members.
    pub mask1: &'a [Vec<usize>],
    pub mask2: &'a [Vec<usize>],
    /// Stripped cluster sequences (member-major: `seq1[i]` is member `i`'s
    /// gapped row over the segment) for the `del` per-pair residue lookup.
    pub seq1: &'a [&'a [u8]],
    pub seq2: &'a [&'a [u8]],
    /// Per-member weights (`eff1[i]`, `eff2[j]`) for the `del` subtraction.
    pub eff1: &'a [f64],
    pub eff2: &'a [f64],
    /// `amino_map[byte] -> alphabet index` (`>= nalpha` = skip, matching
    /// `Profile::from_aligned`'s `idx < nalphabets` filter).
    pub amino_map: &'a [u8],
    pub nalpha: usize,
}

impl<'a> MultiMtx<'a> {
    fn nclass(&self) -> usize {
        self.matrices.len()
    }

    /// A swapped view (cluster 1 ↔ cluster 2). C's `partA__align_variousdist`
    /// computes the vertical-init column with `match_calc_add`/`_del` called
    /// with `cpmx2s`/`cpmx1s`, `seq2`/`seq1`, `eff2`/`eff1`, `masklist2`/
    /// `masklist1` swapped (`partSalignmm.c:1767-1768`). Substitution
    /// matrices are symmetric, so they stay as-is.
    fn swapped(&self) -> MultiMtx<'a> {
        MultiMtx {
            matrices: self.matrices,
            cpmx1s: self.cpmx2s,
            cpmx2s: self.cpmx1s,
            cpmx1s_sparse: self.cpmx2s_sparse,
            cpmx2s_sparse: self.cpmx1s_sparse,
            mask1: self.mask2,
            mask2: self.mask1,
            seq1: self.seq2,
            seq2: self.seq1,
            eff1: self.eff2,
            eff2: self.eff1,
            amino_map: self.amino_map,
            nalpha: self.nalpha,
        }
    }

    /// Vertical-init match: prof2 column 0 vs all `lgth1` prof1 columns.
    /// (C `partSalignmm.c:1767-1768`, the swapped `match_calc_add`/`_del`.)
    pub fn match_col(&self, lgth1: usize) -> Vec<f64> {
        self.swapped().match_row(0, lgth1)
    }

    /// Compute `match[k]` for `k in 0..lgth2`, scoring cluster-1 column `i1`
    /// against every cluster-2 column.
    ///
    /// FP order mirrors C's DP loop exactly (`partSalignmm.c:1777-1781`):
    /// per class `c` (ascending), `match_calc_add` then immediately
    /// `match_calc_del` — adds and dels are INTERLEAVED per class, not
    /// all-adds-then-all-dels (matters for byte-identity, not math).
    pub fn match_row(&self, i1: usize, lgth2: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; lgth2];
        self.match_row_into(i1, &mut out);
        out
    }

    /// Same as [`match_row`] but writes into a caller-provided buffer to
    /// avoid the per-call `vec![0.0; lgth2]` allocation in hot DP loops.
    /// `out.len()` must equal `lgth2`; the buffer is zeroed on entry so the
    /// `+=` accumulation across distance classes starts from zero (matches
    /// the freshly-allocated-Vec behavior of [`match_row`]).
    pub fn match_row_into(&self, i1: usize, out: &mut [f64]) {
        for v in out.iter_mut() { *v = 0.0; }
        let lgth2 = out.len();
        let nalpha = self.nalpha;
        let mut scarr = vec![0.0f64; nalpha];

        for c in 0..self.nclass() {
            let mtx = &self.matrices[c];
            let cpmx1 = &self.cpmx1s[c];
            let cpmx1_sparse = &self.cpmx1s_sparse[c];
            let cpmx2_sparse = &self.cpmx2s_sparse[c];

            // --- match_calc_add(matrices[c], out, cpmx1s[c], cpmx2s[c], i1) ---
            if i1 < cpmx1.len() {
                // scarr[l] = Σ_j mtx[j][l] * cpmx1[i1][j]   (FMA).
                // Sparse: iterate only the non-zero `j` of cpmx1[i1].
                // FP-identical to dense iteration since
                // `x.mul_add(0, s) = s`. For 1-residue clusters, this is
                // O(nalpha) ops total instead of O(nalpha²).
                let sp = &cpmx1_sparse[i1];
                for l in 0..nalpha {
                    let mut s = 0.0f64;
                    for &(j_u8, v) in sp {
                        s = mtx[j_u8 as usize][l].mul_add(v, s);
                    }
                    scarr[l] = s;
                }
                // out[k] += Σ_l scarr[l] * cpmx2[k][l]. Sparse over `l`:
                // skip zero alphabet positions. FP-identical to dense
                // (zero contributions are no-ops).
                for k in 0..lgth2 {
                    let sp2 = &cpmx2_sparse[k];
                    let mut acc = out[k];
                    for &(l_u8, v) in sp2 {
                        acc = scarr[l_u8 as usize].mul_add(v, acc);
                    }
                    out[k] = acc;
                }
            }

            // --- match_calc_del(c): subtract this class's spurious pairs ---
            if !self.mask1[c].is_empty() {
                for k in 0..lgth2 {
                    for m in 0..self.mask1[c].len() {
                        let i = self.mask1[c][m];
                        let j = self.mask2[c][m];
                        let b1 = self.seq1[i][i1];
                        let b2 = self.seq2[j][k];
                        if b1 == b'-' || b2 == b'-' { continue; }
                        let c1 = self.amino_map[b1 as usize] as usize;
                        let c2 = self.amino_map[b2 as usize] as usize;
                        if c1 >= self.nalpha || c2 >= self.nalpha { continue; }
                        out[k] -= mtx[c1][c2] * self.eff1[i] * self.eff2[j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a per-class column profile [col][alpha] from member rows + weights.
    fn build_cpmx(seqs: &[&[u8]], eff: &[f64], amino: &[u8], nalpha: usize, ncol: usize) -> Vec<Vec<f64>> {
        let mut p = vec![vec![0.0f64; nalpha]; ncol];
        for (s, &w) in seqs.iter().zip(eff.iter()) {
            for col in 0..ncol {
                let b = s[col];
                if b == b'-' { continue; }
                let a = amino[b as usize] as usize;
                if a < nalpha { p[col][a] += w; }
            }
        }
        p
    }

    // Naive O(n1*n2) per-pair reference: match[k] = Σ_{i,j} M_{which[i][j]}[res]·eff.
    fn naive_match_row(
        i1: usize, lgth2: usize,
        seq1: &[&[u8]], seq2: &[&[u8]], eff1: &[f64], eff2: &[f64],
        which: &[Vec<usize>], matrices: &[Vec<Vec<f64>>], amino: &[u8], nalpha: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; lgth2];
        for k in 0..lgth2 {
            for i in 0..seq1.len() {
                for j in 0..seq2.len() {
                    let b1 = seq1[i][i1]; let b2 = seq2[j][k];
                    if b1 == b'-' || b2 == b'-' { continue; }
                    let c1 = amino[b1 as usize] as usize; let c2 = amino[b2 as usize] as usize;
                    if c1 >= nalpha || c2 >= nalpha { continue; }
                    let c = which[i][j];
                    out[k] += matrices[c][c1][c2] * eff1[i] * eff2[j];
                }
            }
        }
        out
    }

    #[test]
    fn multimtx_matches_naive_two_classes() {
        // 3 residues A,B,gap. amino: A=0,B=1,'-' handled by caller.
        let nalpha = 2;
        let mut amino = vec![255u8; 256];
        amino[b'A' as usize] = 0;
        amino[b'B' as usize] = 1;
        // 2 clusters: clus1 has 2 seqs, clus2 has 2 seqs, 2 columns.
        let s1a: &[u8] = b"AB"; let s1b: &[u8] = b"BA";
        let s2a: &[u8] = b"AA"; let s2b: &[u8] = b"BB";
        let seq1 = [s1a, s1b]; let seq2 = [s2a, s2b];
        let eff1 = [0.6, 0.4]; let eff2 = [0.7, 0.3];
        // 2 distance classes, distinct matrices.
        let m0 = vec![vec![5.0, -1.0], vec![-1.0, 4.0]];
        let m1 = vec![vec![2.0, 0.0], vec![0.0, 1.0]];
        let matrices = vec![m0, m1];
        // which[i][j]: pair classes. Make a "complex" grouping that triggers masks.
        let which = vec![vec![0usize, 1], vec![1, 0]];
        // classifypairs: eff1s[c][i] = eff1[i] if i in any pair of class c.
        // class0: pairs (0,0),(1,1) -> i in {0,1}, j in {0,1}; class1: (0,1),(1,0) -> same.
        let ncol = 2;
        // eff1s/eff2s per class (mirrors classify_pairs overwrite semantics).
        let mut eff1s = vec![vec![0.0; 2]; 2];
        let mut eff2s = vec![vec![0.0; 2]; 2];
        let mut mask1 = vec![Vec::new(); 2];
        let mut mask2 = vec![Vec::new(); 2];
        for i in 0..2 { for j in 0..2 {
            let c = which[i][j];
            eff1s[c][i] = eff1[i];
            eff2s[c][j] = eff2[j];
        }}
        for c in 0..2 { for i in 0..2 { for j in 0..2 {
            if eff1s[c][i]*eff2s[c][j] != 0.0 && c != which[i][j] {
                mask1[c].push(i); mask2[c].push(j);
            }
        }}}
        let cpmx1s: Vec<Vec<Vec<f64>>> = (0..2).map(|c| build_cpmx(&seq1, &eff1s[c], &amino, nalpha, ncol)).collect();
        let cpmx2s: Vec<Vec<Vec<f64>>> = (0..2).map(|c| build_cpmx(&seq2, &eff2s[c], &amino, nalpha, ncol)).collect();
        let sparsify = |dense: &Vec<Vec<Vec<f64>>>| -> Vec<Vec<Vec<(u8,f64)>>> {
            dense.iter().map(|cls| cls.iter().map(|col| {
                let mut v: Vec<(u8,f64)> = Vec::new();
                for (l, &x) in col.iter().enumerate() { if x != 0.0 { v.push((l as u8, x)); } }
                v
            }).collect()).collect()
        };
        let cpmx1s_sparse = sparsify(&cpmx1s);
        let cpmx2s_sparse = sparsify(&cpmx2s);

        let mm = MultiMtx {
            matrices: &matrices, cpmx1s: &cpmx1s, cpmx2s: &cpmx2s,
            cpmx1s_sparse: &cpmx1s_sparse, cpmx2s_sparse: &cpmx2s_sparse,
            mask1: &mask1, mask2: &mask2,
            seq1: &seq1, seq2: &seq2, eff1: &eff1, eff2: &eff2,
            amino_map: &amino, nalpha,
        };
        for i1 in 0..ncol {
            let got = mm.match_row(i1, ncol);
            let want = naive_match_row(i1, ncol, &seq1, &seq2, &eff1, &eff2, &which, &matrices, &amino, nalpha);
            for k in 0..ncol {
                assert!((got[k]-want[k]).abs() < 1e-12, "i1={i1} k={k} got={} want={}", got[k], want[k]);
            }
        }
    }

    #[test]
    fn multimtx_single_class_equals_plain() {
        // 1 class, no masks -> equals plain cpmx·M·cpmx.
        let nalpha = 2;
        let mut amino = vec![255u8; 256];
        amino[b'A' as usize] = 0; amino[b'B' as usize] = 1;
        let s1a: &[u8] = b"AB"; let s2a: &[u8] = b"BA";
        let seq1 = [s1a]; let seq2 = [s2a];
        let eff1 = [1.0]; let eff2 = [1.0];
        let matrices = vec![vec![vec![5.0, -1.0], vec![-1.0, 4.0]]];
        let which = vec![vec![0usize]];
        let ncol = 2;
        let eff1s = vec![vec![1.0]]; let eff2s = vec![vec![1.0]];
        let cpmx1s: Vec<Vec<Vec<f64>>> = vec![build_cpmx(&seq1, &eff1s[0], &amino, nalpha, ncol)];
        let cpmx2s: Vec<Vec<Vec<f64>>> = vec![build_cpmx(&seq2, &eff2s[0], &amino, nalpha, ncol)];
        let sparsify = |dense: &Vec<Vec<Vec<f64>>>| -> Vec<Vec<Vec<(u8,f64)>>> {
            dense.iter().map(|cls| cls.iter().map(|col| {
                let mut v: Vec<(u8,f64)> = Vec::new();
                for (l, &x) in col.iter().enumerate() { if x != 0.0 { v.push((l as u8, x)); } }
                v
            }).collect()).collect()
        };
        let cpmx1s_sparse = sparsify(&cpmx1s);
        let cpmx2s_sparse = sparsify(&cpmx2s);
        let mask1 = vec![Vec::new()]; let mask2 = vec![Vec::new()];
        let mm = MultiMtx {
            matrices: &matrices, cpmx1s: &cpmx1s, cpmx2s: &cpmx2s,
            cpmx1s_sparse: &cpmx1s_sparse, cpmx2s_sparse: &cpmx2s_sparse,
            mask1: &mask1, mask2: &mask2,
            seq1: &seq1, seq2: &seq2, eff1: &eff1, eff2: &eff2,
            amino_map: &amino, nalpha,
        };
        let got = mm.match_row(0, ncol);
        let want = naive_match_row(0, ncol, &seq1, &seq2, &eff1, &eff2, &which, &matrices, &amino, nalpha);
        for k in 0..ncol {
            assert!((got[k]-want[k]).abs() < 1e-12, "k={k} got={} want={}", got[k], want[k]);
        }
    }
}
