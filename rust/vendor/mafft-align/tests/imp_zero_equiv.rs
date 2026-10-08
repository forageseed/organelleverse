//! Sanity: profile_align_imp(impmtx = zeros) must equal profile_align.

use mafft_align::{Profile, profile_align, profile_align_imp, GapModel};

fn build_amino_map() -> [u8; 256] {
    let mut map = [0xFFu8; 256];
    for (i, c) in b"ACDEFGHIKLMNPQRSTVWY".iter().enumerate() {
        map[*c as usize] = i as u8;
    }
    map[b'-' as usize] = 20;
    map
}

fn build_simple_matrix() -> Vec<Vec<f64>> {
    let n = 26;
    let mut m = vec![vec![-1.0f64; n]; n];
    for i in 0..20 {
        m[i][i] = 4.0;
    }
    m
}

#[test]
fn imp_zero_matches_profile_align() {
    let map = build_amino_map();
    let mtx = build_simple_matrix();
    let g1: Vec<&[u8]> = vec![b"ACDEFGHIKLM"];
    let g2: Vec<&[u8]> = vec![b"ACDE-GHIKLM"];
    let prof1 = Profile::from_aligned(&g1, &[1.0], &map, 26);
    let prof2 = Profile::from_aligned(&g2, &[1.0], &map, 26);
    let gap = GapModel::new(-100.0, -10.0);

    let plain = profile_align(&prof1, &prof2, &mtx, &gap, false, false);

    let imp_zero = vec![vec![0.0f64; prof2.length]; prof1.length];
    let with_imp = profile_align_imp(
        &prof1, &prof2, &mtx, &gap, false, false, Some(&imp_zero),
    );

    assert_eq!(plain.operations, with_imp.operations,
        "ops differ: plain={:?} imp={:?}", plain.operations, with_imp.operations);
    assert!((plain.score - with_imp.score).abs() < 1e-9,
        "score differs: plain={} imp={}", plain.score, with_imp.score);
}
