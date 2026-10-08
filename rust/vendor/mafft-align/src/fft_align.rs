/// FFT-accelerated alignment.
///
/// Ports the C `Falign()` from Falign.c.
///
/// Uses multi-channel FFT cross-correlation to find anchor points between
/// two sequence groups, selects optimal anchors via DP, then applies full
/// DP alignment within each anchored segment.

use mafft_fft::{
    alignable_segments, block_align, get_top_candidates,
    multichannel_correlate, SegmentParams,
};

use crate::dp::{Alignment, GapModel};
use crate::profile::{align_with_anchors_outgap, profile_align, Profile};

/// Parameters controlling FFT-accelerated alignment.
#[derive(Debug, Clone)]
pub struct FftAlignParams {
    /// Number of candidate lags to evaluate.
    pub num_candidates: usize,
    /// Segment detection parameters.
    pub segment_params: SegmentParams,
    /// Gap model for DP alignment within segments.
    pub gap: GapModel,
    /// Whether to penalize head/tail gaps.
    pub head_gap: bool,
    pub tail_gap: bool,
    /// Number of FFT channels (20 for protein, 4 for DNA). Ignored when
    /// `property_channels` is `Some(_)`.
    pub num_channels: usize,
    /// Per-internal-index polarity and volume values. When `Some`, the FFT
    /// uses 2-channel polarity+volume convolution mirroring C's `seq_vec_2`
    /// path (`Falign.c:342-348`, taken when `fftscore && scoremtx != -1`).
    /// When `None`, falls back to `num_channels`-indicator channels via
    /// `seq_vec_3`. C uses property channels for all protein scoring
    /// matrices (BLOSUM/JTT/TM); DNA always uses indicator channels.
    pub property_channels: Option<(Vec<f64>, Vec<f64>)>,
}

impl FftAlignParams {
    pub fn protein() -> Self {
        Self {
            num_candidates: 20,
            segment_params: SegmentParams::protein(),
            gap: GapModel::default(),
            head_gap: true,
            tail_gap: true,
            num_channels: 20,
            property_channels: None,
        }
    }

    pub fn dna() -> Self {
        Self {
            num_candidates: 20,
            segment_params: SegmentParams::dna(),
            gap: GapModel::default(),
            head_gap: true,
            tail_gap: true,
            num_channels: 4,
            property_channels: None,
        }
    }
}

/// An anchor point between two profiles.
#[derive(Debug, Clone)]
pub struct Anchor {
    pub pos1: usize,
    pub pos2: usize,
}

/// Find FFT-based anchor points between two profiles.
///
/// Performs the FFT correlation, segment detection, and anchor selection
/// steps of Falign, returning the anchors without running the DP.
/// Returns `None` if no good anchors are found.
pub fn find_fft_anchors(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    params: &FftAlignParams,
) -> Option<Vec<(usize, usize)>> {
    let n = prof1.length;
    let m = prof2.length;

    if n == 0 || m == 0 {
        return None;
    }

    let fft_size = n.max(m).next_power_of_two();
    let (channels_a, channels_b) = if let Some((polarity, volume)) = &params.property_channels {
        // Mirrors C's `seq_vec_2` 2-channel polarity+volume convolution
        // (`Falign.c:342-348`). Used by all protein scoring matrices.
        (
            profile_to_property_channels(prof1, polarity, volume, fft_size),
            profile_to_property_channels(prof2, polarity, volume, fft_size),
        )
    } else {
        (
            profile_to_channels(prof1, params.num_channels, fft_size),
            profile_to_channels(prof2, params.num_channels, fft_size),
        )
    };
    let raw_corr = multichannel_correlate(&channels_a, &channels_b);

    let nlen = fft_size;
    let nlen2 = nlen / 2;
    let mut soukan = vec![0.0f64; nlen];
    for idx in 0..nlen {
        let lag = idx as i32 - nlen2 as i32;
        let src = lag.rem_euclid(nlen as i32) as usize;
        soukan[idx] = raw_corr[src];
    }

    let candidates = get_top_candidates(&soukan, params.num_candidates);

    // Mirrors C's `Falign` (`Falign.c:1307-1372`): accumulate segments from
    // candidate lags into a single flat list (sorted by score), then
    // independently sort centers in each seq dimension and run
    // `blockAlign2` over a sparse cross-score matrix. The DP picks the
    // best non-conflicting subset of anchors across all candidate lags.
    //
    // C's loop bounds: `if (lag <= -len1 || lag >= len2) continue;`
    // (`Falign.c:1310`) and break on first empty `tmpint == 0`
    // (`Falign.c:1330`).
    struct PairSeg { center1: usize, center2: usize, score: f64 }
    let mut all: Vec<PairSeg> = Vec::new();
    for cand in &candidates {
        let lag = cand.lag;
        if lag <= -(n as i32) || lag >= m as i32 { continue; }
        let shifted_scores = shift_and_score(prof1, prof2, matrix, lag);
        let segments = alignable_segments(&shifted_scores, &params.segment_params);
        if segments.is_empty() {
            break; // C: `if(tmpint == 0) break;`
        }
        for seg in segments {
            // shift_and_score's `seg.center` is in the shifted-frame:
            //   lag >= 0: index in prof1 → seq1_idx = center, seq2_idx = center + lag
            //   lag <  0: index in prof2 → seq2_idx = center, seq1_idx = center - lag
            // Either way `seq2_idx - seq1_idx == lag`. Mirrors C's segment1/2
            // mapping in Falign.c:540-563.
            let (c1, c2_signed): (usize, i32) = if lag >= 0 {
                (seg.center, seg.center as i32 + lag)
            } else {
                ((seg.center as i32 - lag) as usize, seg.center as i32)
            };
            if c1 >= n || c2_signed < 0 || (c2_signed as usize) >= m { continue; }
            all.push(PairSeg { center1: c1, center2: c2_signed as usize, score: seg.score });
        }
    }

    if all.is_empty() { return None; }

    let nseg = all.len();
    let mut sort1: Vec<usize> = (0..nseg).collect();
    sort1.sort_by(|&a, &b| all[a].center1.cmp(&all[b].center1));
    let mut sort2: Vec<usize> = (0..nseg).collect();
    sort2.sort_by(|&a, &b| all[a].center2.cmp(&all[b].center2));

    let mut rank1 = vec![0usize; nseg];
    let mut rank2 = vec![0usize; nseg];
    for (r, &i) in sort1.iter().enumerate() { rank1[i] = r; }
    for (r, &i) in sort2.iter().enumerate() { rank2[i] = r; }

    let size = nseg + 2;
    let mut crossscore = vec![vec![0.0f64; size]; size];
    for i in 0..nseg {
        crossscore[rank1[i] + 1][rank2[i] + 1] = all[i].score;
    }
    crossscore[0][0] = 1e7;
    crossscore[size - 1][size - 1] = 1e7;

    let (sel_i, sel_j) = block_align(&crossscore, params.gap.open);

    let mut anchors: Vec<(usize, usize)> = Vec::new();
    for (&si, &sj) in sel_i.iter().zip(sel_j.iter()) {
        if si == 0 || si == size - 1 { continue; }
        if sj == 0 || sj == size - 1 { continue; }
        let r1 = si - 1;
        let r2 = sj - 1;
        if r1 >= nseg || r2 >= nseg { continue; }
        let seg_idx = sort1[r1];
        if rank2[seg_idx] != r2 { continue; }
        let seg = &all[seg_idx];
        if seg.center1 < n && seg.center2 < m {
            anchors.push((seg.center1, seg.center2));
        }
    }

    if anchors.is_empty() {
        None
    } else {
        Some(anchors)
    }
}

/// Perform FFT-accelerated profile alignment.
///
/// 1. Converts profiles to per-residue-type complex vectors.
/// 2. Uses multi-channel FFT cross-correlation to find the best lags.
/// 3. Detects alignable segments at each candidate lag.
/// 4. Selects optimal non-overlapping anchor pairs via DP (block_align).
/// 5. Runs full DP alignment within each segment.
///
/// Falls back to direct profile DP if no good anchors are found.
pub fn fft_profile_align(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    params: &FftAlignParams,
) -> Alignment {
    if prof1.length == 0 || prof2.length == 0 {
        return Alignment {
            seq1: Vec::new(),
            seq2: Vec::new(),
            score: 0.0,
            operations: Vec::new(),
        };
    }

    match find_fft_anchors(prof1, prof2, matrix, params) {
        Some(anchors) => align_with_anchors_outgap(
            prof1, prof2, matrix, &params.gap, &anchors,
            // The fft path's `head_gap`/`tail_gap` are conventionally
            // both equal to the outer C `outgap` (Falign passes a single
            // value into its first/last segment). Pass `head_gap` as the
            // canonical outgap here — for our progressive merges
            // `head_gap == tail_gap` (both controlled by
            // `penalize_term_gaps` in `progressive::merge_step_cached`).
            params.head_gap,
        ),
        None => profile_align(prof1, prof2, matrix, &params.gap, params.head_gap, params.tail_gap),
    }
}

/// Convert profile frequencies to per-channel complex vectors for FFT.
fn profile_to_channels(
    prof: &Profile,
    num_channels: usize,
    fft_size: usize,
) -> Vec<Vec<num_complex::Complex64>> {
    let mut channels =
        vec![vec![num_complex::Complex64::new(0.0, 0.0); fft_size]; num_channels];
    for pos in 0..prof.length.min(fft_size) {
        for ch in 0..num_channels.min(prof.nalphabets) {
            channels[ch][pos] = num_complex::Complex64::new(prof.freqs[pos][ch], 0.0);
        }
    }
    channels
}

/// Convert profile frequencies to 2-channel polarity+volume FFT inputs.
///
/// Mirrors C's `seq_vec_2` (`Falign.c:43-54`) but in profile space: for
/// each position `p`, channel 0 = `Σ_a freqs[p][a] · polarity[a]`, channel
/// 1 = `Σ_a freqs[p][a] · volume[a]`. Imaginary part is zero (C only sets
/// `result->R`).
///
/// `polarity[a]` and `volume[a]` are indexed by INTERNAL alphabet index
/// (0..nscored), already mapped from C's character-indexed `polarity[256]`
/// / `volume[256]` arrays via the scoring context's amino-map.
fn profile_to_property_channels(
    prof: &Profile,
    polarity: &[f64],
    volume: &[f64],
    fft_size: usize,
) -> Vec<Vec<num_complex::Complex64>> {
    use num_complex::Complex64;
    let mut channels = vec![vec![Complex64::new(0.0, 0.0); fft_size]; 2];
    let nalpha = prof.nalphabets.min(polarity.len()).min(volume.len());
    for pos in 0..prof.length.min(fft_size) {
        let mut p_val = 0.0f64;
        let mut v_val = 0.0f64;
        for a in 0..nalpha {
            let f = prof.freqs[pos][a];
            if f != 0.0 {
                // FMA matches gcc -O3's fusion of `a + b*c` (`Falign.c:seq_vec_2`).
                p_val = f.mul_add(polarity[a], p_val);
                v_val = f.mul_add(volume[a], v_val);
            }
        }
        channels[0][pos].re = p_val;
        channels[1][pos].re = v_val;
    }
    channels
}

/// Compute per-position match scores between two profiles at a given lag.
///
/// Mirrors C's `zurasu2` + `alignableReagion` per-position scoring. The
/// scoring frame is the LATER-starting sequence: for `lag >= 0` we pair
/// `(prof1[i], prof2[i + lag])`; for `lag < 0` we pair
/// `(prof1[i - lag], prof2[i])`. Either way `prof2_idx - prof1_idx = lag`.
///
/// The result length matches C's `MIN(strlen(aseq1[0]), strlen(aseq2[0]))`
/// after `zurasu2` advances the pointer of the earlier-starting sequence:
///   `lag >= 0`: `min(prof1.length, prof2.length - lag)`
///   `lag <  0`: `min(prof1.length + lag, prof2.length)`
/// scores beyond that range are absent — segment detection sees only valid
/// overlap, not zero-padding.
fn shift_and_score(
    prof1: &Profile,
    prof2: &Profile,
    matrix: &[Vec<f64>],
    lag: i32,
) -> Vec<f64> {
    let n = prof1.length as i32;
    let m = prof2.length as i32;
    let valid_len = if lag >= 0 {
        n.min(m - lag).max(0)
    } else {
        (n + lag).min(m).max(0)
    } as usize;
    let mut scores = vec![0.0f64; valid_len];
    if lag >= 0 {
        for i in 0..valid_len {
            scores[i] = prof1.match_score(i, prof2, i + lag as usize, matrix);
        }
    } else {
        let off = (-lag) as usize;
        for i in 0..valid_len {
            scores[i] = prof1.match_score(i + off, prof2, i, matrix);
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_setup() -> (Vec<Vec<f64>>, [u8; 256], usize) {
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
        (mtx, map, 5)
    }

    #[test]
    fn fft_align_identical_profiles() {
        let (mtx, map, nalpha) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGTACGTACGTACGTACGTACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, nalpha);
        let params = FftAlignParams {
            num_candidates: 5,
            segment_params: SegmentParams {
                window_size: 4,
                threshold: 100.0,
                max_segment_size: 50,
            },
            gap: GapModel::new(-200.0, -10.0),
            head_gap: true,
            tail_gap: true,
            num_channels: 4,
            property_channels: None,
        };
        let aln = fft_profile_align(&prof, &prof, &mtx, &params);
        assert!(aln.score > 0.0);
        assert!(!aln.operations.is_empty());
    }

    #[test]
    fn fft_align_falls_back_on_short() {
        let (mtx, map, nalpha) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGT"];
        let prof = Profile::from_aligned(&seqs, &[1.0], &map, nalpha);
        let params = FftAlignParams::protein();
        let aln = fft_profile_align(&prof, &prof, &mtx, &params);
        assert!((aln.score - 400.0).abs() < 1e-6);
    }
}
