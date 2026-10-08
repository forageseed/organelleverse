/// Constrained alignment: FFT alignment guided by local homology tables
/// and partial (segment-wise) profile alignment.
///
/// Ports the C `Falign_localhom.c` and `partSalignmm.c`.

use mafft_types::LocalHomologyTable;
use mafft_fft::{SegmentParams, alignable_segments, block_align};

use crate::dp::{Alignment, GapModel};
use crate::profile::{align_with_anchors, profile_align, Profile};

/// Parameters for constrained FFT alignment.
#[derive(Debug, Clone)]
pub struct ConstrainedAlignParams {
    pub gap: GapModel,
    pub segment_params: SegmentParams,
    /// Weight given to local homology importance scores when selecting anchors.
    pub constraint_weight: f64,
}

impl Default for ConstrainedAlignParams {
    fn default() -> Self {
        Self {
            gap: GapModel::default(),
            segment_params: SegmentParams::protein(),
            constraint_weight: 1.0,
        }
    }
}

/// Perform FFT alignment with local homology constraints.
///
/// Extends `fft_profile_align` by weighting the anchor cross-scores
/// with importance values from the `LocalHomologyTable`. Anchors that fall
/// within high-importance local homology regions are preferred.
///
/// Used by L-INS-i and E-INS-i for constraint-guided progressive alignment.
pub fn constrained_profile_align(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    constraints: &LocalHomologyTable,
    group1_members: &[usize],
    group2_members: &[usize],
    params: &ConstrainedAlignParams,
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

    // Build constraint importance map
    let importance_map = build_importance_map(
        constraints, group1_members, group2_members, n, m,
    );

    // Compute site scores with constraint bonus
    let site_scores: Vec<f64> = (0..n)
        .map(|i| {
            if i < m {
                let base = prof1.match_score(i, prof2, i, matrix);
                let bonus = importance_map.get(i).and_then(|row| row.get(i)).copied().unwrap_or(0.0);
                base + bonus * params.constraint_weight
            } else {
                0.0
            }
        })
        .collect();

    // Detect segments
    let segments = alignable_segments(&site_scores, &params.segment_params);

    if segments.is_empty() {
        return profile_align(prof1, prof2, matrix, &params.gap, true, true);
    }

    // Build cross-score matrix for segment pairs
    let nseg = segments.len();
    let mut cross_scores = vec![vec![0.0f64; nseg]; nseg];
    for (i, seg_i) in segments.iter().enumerate() {
        for (j, seg_j) in segments.iter().enumerate() {
            cross_scores[i][j] = seg_i.score.min(seg_j.score);
            let center_i = seg_i.center.min(n - 1);
            let center_j = seg_j.center.min(m - 1);
            if let Some(imp) = importance_map.get(center_i).and_then(|row| row.get(center_j)) {
                cross_scores[i][j] += imp * params.constraint_weight;
            }
        }
    }

    // Select optimal anchor pairs
    let (sel_i, sel_j) = block_align(&cross_scores, params.gap.open);

    if sel_i.is_empty() {
        return profile_align(prof1, prof2, matrix, &params.gap, true, true);
    }

    // Convert to anchor positions and use shared alignment function
    let anchors: Vec<(usize, usize)> = sel_i
        .iter()
        .zip(sel_j.iter())
        .map(|(&si, &sj)| (segments[si].center.min(n - 1), segments[sj].center.min(m - 1)))
        .collect();

    align_with_anchors(prof1, prof2, matrix, &params.gap, &anchors)
}

/// Partial (segment-wise) profile alignment.
///
/// Aligns a sub-region of two profiles, respecting boundary constraints.
/// Ports the core idea from `partSalignmm.c`.
pub fn partial_profile_align(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    gap: &GapModel,
    start1: usize,
    end1: usize,
    start2: usize,
    end2: usize,
) -> Alignment {
    let sub1 = prof1.sub_profile(start1, end1);
    let sub2 = prof2.sub_profile(start2, end2);
    profile_align(&sub1, &sub2, matrix, gap, start1 == 0, end1 == prof1.length)
}

/// Build a sparse importance map from local homology constraints.
fn build_importance_map(
    constraints: &LocalHomologyTable,
    group1: &[usize],
    group2: &[usize],
    len1: usize,
    len2: usize,
) -> Vec<Vec<f64>> {
    let mut map = vec![vec![0.0f64; len2]; len1];

    for &s1 in group1 {
        for &s2 in group2 {
            let regions = constraints.get(s1, s2);
            for region in regions {
                let i_start = (region.start1 as usize).min(len1);
                let i_end = (region.end1 as usize).min(len1);
                let j_start = (region.start2 as usize).min(len2);
                let j_end = (region.end2 as usize).min(len2);

                let area = ((i_end - i_start) * (j_end - j_start)).max(1) as f64;
                let imp_per_cell = region.importance / area;

                for i in i_start..i_end {
                    for j in j_start..j_end {
                        map[i][j] += imp_per_cell;
                    }
                }
            }
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use mafft_types::LocalHomologyTable;

    fn simple_setup() -> (Vec<Vec<f64>>, [u8; 256], usize) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 { mtx[i][i] = 100.0; }
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0; map[b'C' as usize] = 1;
        map[b'G' as usize] = 2; map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map, 5)
    }

    #[test]
    fn constrained_align_with_empty_constraints() {
        let (mtx, map, nalpha) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, nalpha);
        let table = LocalHomologyTable::new(2);
        let params = ConstrainedAlignParams::default();

        let aln = constrained_profile_align(
            &prof, &prof, &mtx, &table, &[0], &[1], &params,
        );
        assert!(aln.score > 0.0 || !aln.operations.is_empty());
    }

    #[test]
    fn partial_profile_align_subregion() {
        let (mtx, map, nalpha) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, nalpha);
        let gap = GapModel::new(-200.0, -10.0);

        let aln = partial_profile_align(&prof, &prof, &mtx, &gap, 2, 6, 2, 6);
        assert_eq!(aln.operations.len(), 4);
    }
}
