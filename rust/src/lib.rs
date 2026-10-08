//! OrganelleVerse Rust acceleration kernel.
//!
//! Three hot-path functions that benefit from Rust's speed:
//!   1. `kmer_overlap`  — k-mer overlap fragment detection (transfer/MTPT/NUMT/HGT)
//!   2. `erc_correlation` — ERC2 all-by-all residual correlation (coevolution)
//!   3. `kmer_jaccard`  — Jaccard similarity via k-mer sets (barcode/pangenome)
//!
//! All return plain Python types (dicts, tuples) via pyo3 so the Python side
//! can consume them without a shared schema.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;
use std::collections::HashSet;

mod cm;

// =========================================================================
// 1. K-MER OVERLAP DETECTION
//    Scans a target sequence for runs of k-mers shared with a source,
//    returning matched fragments (start, end). Both strands of the source
//    are indexed (forward k-mers + their reverse complements) so reverse-
//    orientation integrations (common for MTPT/NUMT) are detected. Uses
//    2-bit encoding + HashSet.
//    Returns 1-based inclusive [start, end] where end is the last base
//    covered by the last matching k-mer (tail run uses the same formula).
// =========================================================================

fn encode_kmers(seq: &[u8], k: usize) -> HashSet<u64> {
    let mut set = HashSet::with_capacity(seq.len());
    if seq.len() < k {
        return set;
    }
    for i in 0..=seq.len() - k {
        let window = &seq[i..i + k];
        if window.iter().all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T')) {
            let code = encode_2bit(window);
            set.insert(code);
            // Also index the reverse complement, so reverse-orientation
            // integrations (common for MTPT/NUMT) are detected.
            let rc_code = encode_2bit_rc(window);
            set.insert(rc_code);
        }
    }
    set
}

/// Strand-specific k-mer encoding for Jaccard similarity (barcode-style):
/// forward k-mers only, windows containing non-ACGT (e.g. N) are skipped,
/// input is upper-cased first — exactly the semantics of the Python
/// barcode reference `compute_jaccard_identity`. Folding reverse
/// complements here (as `encode_kmers` does for MTPT/NUMT overlap)
/// would make AAAA vs TTTT self-similar, which is wrong for a
/// strand-specific barcode identity.
fn encode_kmers_stranded(seq: &[u8], k: usize) -> HashSet<u64> {
    let upper: Vec<u8> = seq.iter().map(|&b| b.to_ascii_uppercase()).collect();
    let mut set = HashSet::with_capacity(upper.len());
    if upper.len() < k {
        return set;
    }
    for i in 0..=upper.len() - k {
        let window = &upper[i..i + k];
        if window.iter().all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T')) {
            set.insert(encode_2bit(window));
        }
    }
    set
}

#[inline]
fn encode_2bit(seq: &[u8]) -> u64 {
    let mut code: u64 = 0;
    for &b in seq {
        let bits = match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 4, // should not happen (filtered above)
        };
        code = (code << 2) | bits;
    }
    code
}

/// 2-bit encoding of the reverse complement of ``seq`` (A<->T, C<->G, then reversed).
#[inline]
fn encode_2bit_rc(seq: &[u8]) -> u64 {
    let mut code: u64 = 0;
    for &b in seq.iter().rev() {
        let bits = match b {
            b'A' => 3, // complement of A is T (=3)
            b'C' => 2, // complement of C is G (=2)
            b'G' => 1, // complement of G is C (=1)
            b'T' => 0, // complement of T is A (=0)
            _ => 4,
        };
        code = (code << 2) | bits;
    }
    code
}

fn detect_fragments(target: &[u8], source_kmers: &HashSet<u64>, k: usize, min_len: usize) -> Vec<(usize, usize)> {
    let mut fragments = Vec::new();
    if target.len() < k {
        return fragments;
    }
    let mut run_start: Option<usize> = None;
    let mut last_match_end: usize = 0; // 0-based exclusive end of the run
    for i in 0..=target.len() - k {
        let window = &target[i..i + k];
        let is_match = if window.iter().all(|&b| matches!(b, b'A' | b'C' | b'G' | b'T')) {
            source_kmers.contains(&encode_2bit(window))
        } else {
            false
        };
        if is_match {
            if run_start.is_none() {
                run_start = Some(i);
            }
            last_match_end = i + k; // 0-based exclusive
        } else if let Some(start) = run_start {
            let run_len = last_match_end - start;
            if run_len >= min_len {
                // 1-based inclusive: [start+1, last_match_end]
                fragments.push((start + 1, last_match_end));
            }
            run_start = None;
        }
    }
    // Tail run uses the identical formula as in-loop runs.
    if let Some(start) = run_start {
        let run_len = last_match_end - start;
        if run_len >= min_len {
            fragments.push((start + 1, last_match_end));
        }
    }
    fragments
}

/// Python: kmer_overlap(target_seq: str, source_seq: str, k: int, min_len: int) -> list[tuple[int,int]]
#[pyfunction]
fn kmer_overlap(target_seq: &str, source_seq: &str, k: usize, min_len: usize) -> Vec<(usize, usize)> {
    let target = target_seq.as_bytes();
    let source = source_seq.as_bytes();
    let source_kmers = encode_kmers(source, k);
    detect_fragments(target, &source_kmers, k, min_len)
}

// =========================================================================
// 2. ERC2 ALL-BY-ALL CORRELATION (the biggest bottleneck: O(N²))
//    Input: gene_names + branch_length matrix (genes × branches)
//    Steps: transform → trimmed-mean master → OLS residuals →
//           pairwise Pearson + Spearman → Benjamini-Hochberg FDR
// =========================================================================

fn transform_bl(vec: &[f64], method: &str) -> Vec<f64> {
    match method {
        "sqrt" => vec.iter().map(|&v| if v > 0.0 { v.sqrt() } else { 0.0 }).collect(),
        "log" => {
            let min_pos = vec.iter().filter(|&&v| v > 0.0).cloned().fold(f64::MAX, f64::min);
            let offset = if min_pos == f64::MAX { 1e-6 } else { min_pos };
            vec.iter().map(|&v| if v >= 0.0 { (v + offset).ln() } else { 0.0 }).collect()
        }
        _ => vec.to_vec(),
    }
}

fn compute_master(genes: &[Vec<f64>], trim_prop: f64) -> Vec<f64> {
    let n_genes = genes.len();
    if n_genes == 0 {
        return vec![];
    }
    let n_branches = genes.iter().map(|g| g.len()).min().unwrap_or(0);
    let mut master = vec![f64::NAN; n_branches];
    for j in 0..n_branches {
        let mut col: Vec<f64> = genes.iter()
            .filter_map(|g| {
                let v = g.get(j).copied().unwrap_or(f64::NAN);
                if v.is_nan() { None } else { Some(v) }
            })
            .collect();
        if col.is_empty() {
            continue;
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = col.len();
        let trim = (n as f64 * trim_prop) as usize;
        let trimmed = if n > 2 * trim { &col[trim..n - trim] } else { &col[..] };
        master[j] = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
    }
    master
}

fn compute_residuals(vec: &[f64], master: &[f64]) -> Vec<f64> {
    let n = vec.len().min(master.len());
    // valid positions: neither master nor vec is NaN
    let valid: Vec<usize> = (0..n)
        .filter(|&i| !master[i].is_nan() && !vec[i].is_nan())
        .collect();
    if valid.len() < 2 {
        return vec![f64::NAN; n];
    }
    let pairs: Vec<(f64, f64)> = valid.iter().map(|&i| (master[i], vec[i])).collect();
    let mx: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64;
    let my: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64;
    let num: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    let slope = if den != 0.0 { num / den } else { 0.0 };
    let intercept = my - slope * mx;
    let mut result = vec![f64::NAN; n];
    for &i in &valid {
        result[i] = vec[i] - (slope * master[i] + intercept);
    }
    result
}

fn pearson(x: &[f64], y: &[f64]) -> (f64, usize) {
    // overlap where neither is NaN (missing data)
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return (0.0, n);
    }
    let mx: f64 = pairs.iter().map(|p| p.0).sum::<f64>() / n as f64;
    let my: f64 = pairs.iter().map(|p| p.1).sum::<f64>() / n as f64;
    let num: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let da: f64 = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f64>().sqrt();
    let db: f64 = pairs.iter().map(|(_, y)| (y - my).powi(2)).sum::<f64>().sqrt();
    let r = if da != 0.0 && db != 0.0 { num / (da * db) } else { 0.0 };
    (r, n)
}

fn rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && indexed[j + 1].1 == indexed[i].1 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();
    if pairs.len() < 2 {
        return 0.0;
    }
    let rx = rank(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let ry = rank(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    let (r, _) = pearson(&rx, &ry);
    r
}

fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = x.iter().zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();
    let n = pairs.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut tie_x = 0i64;
    let mut tie_y = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pairs[i].0 - pairs[j].0;
            let dy = pairs[i].1 - pairs[j].1;
            if dx == 0.0 || dy == 0.0 {
                if dx == 0.0 { tie_x += 1; }
                if dy == 0.0 { tie_y += 1; }
            } else if (dx > 0.0) == (dy > 0.0) {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let n_pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    let v0 = n_pairs - tie_x as f64;
    let v1 = n_pairs - tie_y as f64;
    let denom = (v0 * v1).sqrt();
    if denom > 0.0 { (concordant - discordant) as f64 / denom } else { 0.0 }
}

fn pearson_pvalue(r: f64, n: usize) -> f64 {
    if n <= 2 {
        return 1.0;
    }
    if r.abs() >= 1.0 {
        return 0.0;
    }
    let df = (n - 2) as f64;
    let t2 = r * r * df / (1.0 - r * r);
    if !t2.is_finite() {
        return 0.0;
    }
    // x = df / (df + t²), p = I_x(df/2, 0.5) via incomplete beta
    let x = df / (df + t2);
    let p_one_tail = 0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5);
    (2.0 * p_one_tail).min(1.0)
}

/// Regularized incomplete beta via continued fraction (Lentz).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lbeta).exp() / a;
    front * beta_cf(x, a, b)
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny { d = tiny; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200i32 {
        let m2 = 2.0 * m as f64;
        let mut aa = m as f64 * (b - m as f64) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + aa / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m as f64) * (qab + m as f64) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + aa / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 { break; }
    }
    h
}

/// Lanczos approximation of log-gamma.
fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        return ((1.0 - x) * std::f64::consts::PI).sin().ln() - ln_gamma(1.0 - x) - std::f64::consts::LN_2;
    }
    let g = 7.0;
    let p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut a = p[0];
    let t = x + g + 0.5;
    for (i, &pi) in p.iter().enumerate().skip(1) {
        a += pi / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn apply_fdr(pairs: &mut [ErcPair], threshold: f64) {
    // sort by pval ascending
    pairs.sort_by(|a, b| a.pval.partial_cmp(&b.pval).unwrap());
    let m = pairs.len() as f64;
    let mut prev = 1.0;
    for i in (0..pairs.len()).rev() {
        let rank = (i + 1) as f64;
        let fdr = (pairs[i].pval * m / rank).min(prev);
        pairs[i].fdr_pval = fdr.min(1.0);
        pairs[i].significant = pairs[i].fdr_pval < threshold;
        prev = pairs[i].fdr_pval;
    }
}

#[derive(Clone)]
struct ErcPair {
    gene_a: usize,
    gene_b: usize,
    r: f64,
    rho: f64,
    tau: f64,
    n: usize,
    pval: f64,
    fdr_pval: f64,
    significant: bool,
}

/// Python: erc_correlation(gene_names, branch_lengths, method, transform, min_overlap, fdr_threshold)
/// Returns a list of dicts.
#[pyfunction]
fn erc_correlation(
    py: Python<'_>,
    gene_names: Vec<String>,
    branch_lengths: Vec<Vec<f64>>,
    method: &str,
    transform: &str,
    min_overlap: usize,
    fdr_threshold: f64,
) -> PyResult<Vec<PyObject>> {
    let n_genes = gene_names.len();
    if n_genes < 2 {
        return Ok(vec![]);
    }

    // Step 1: transform
    let transformed: Vec<Vec<f64>> = if method == "residual" || transform != "none" {
        branch_lengths.iter().map(|g| transform_bl(g, transform)).collect()
    } else {
        branch_lengths.clone()
    };

    // Step 2: master (trimmed mean)
    let master = compute_master(&transformed, 0.05);

    // Step 3: residuals or raw
    let vectors: Vec<Vec<f64>> = if method == "residual" {
        transformed.iter().map(|g| compute_residuals(g, &master)).collect()
    } else {
        transformed.clone()
    };

    // Step 4: all-by-all correlation (parallelized with rayon)
    let indices: Vec<(usize, usize)> = (0..n_genes)
        .flat_map(|i| (i + 1..n_genes).map(move |j| (i, j)))
        .collect();

    let pairs_vec: Vec<Option<ErcPair>> = indices.par_iter().map(|&(i, j)| {
        let (r, n) = pearson(&vectors[i], &vectors[j]);
        if n >= min_overlap {
            let rho = spearman(&vectors[i], &vectors[j]);
            let tau = kendall_tau(&vectors[i], &vectors[j]);
            let pval = pearson_pvalue(r, n);
            Some(ErcPair {
                gene_a: i, gene_b: j,
                r, rho, tau, n, pval,
                fdr_pval: 1.0, significant: false,
            })
        } else {
            None
        }
    }).collect();

    let mut pairs: Vec<ErcPair> = pairs_vec.into_iter().flatten().collect();

    // Step 5: FDR
    apply_fdr(&mut pairs, fdr_threshold);

    // Build Python dicts
    let results: Vec<PyObject> = pairs.into_iter().map(|p| {
        let dict = PyDict::new_bound(py);
        dict.set_item("gene_a", &gene_names[p.gene_a]).unwrap();
        dict.set_item("gene_b", &gene_names[p.gene_b]).unwrap();
        dict.set_item("r", p.r.round_to(4)).unwrap();
        dict.set_item("rho", p.rho.round_to(4)).unwrap();
        dict.set_item("tau", p.tau.round_to(4)).unwrap();
        dict.set_item("n_branches", p.n).unwrap();
        dict.set_item("pval", p.pval).unwrap();
        dict.set_item("fdr_pval", p.fdr_pval).unwrap();
        dict.set_item("significant", p.significant).unwrap();
        dict.into()
    }).collect();

    Ok(results)
}

trait RoundTo {
    fn round_to(&self, decimals: i32) -> f64;
}
impl RoundTo for f64 {
    fn round_to(&self, decimals: i32) -> f64 {
        let factor = 10f64.powi(decimals);
        (self * factor).round() / factor
    }
}

// =========================================================================
// 3. K-MER JACCARD (for barcode/pangenome quick similarity)
// =========================================================================

/// Python: kmer_jaccard(seq_a: str, seq_b: str, k: int) -> f64
/// Strand-specific Jaccard over forward k-mers (barcode semantics; see
/// `encode_kmers_stranded`). Returns 0.0 when either set is empty.
#[pyfunction]
fn kmer_jaccard(seq_a: &str, seq_b: &str, k: usize) -> f64 {
    let kmers_a = encode_kmers_stranded(seq_a.as_bytes(), k);
    let kmers_b = encode_kmers_stranded(seq_b.as_bytes(), k);
    let inter = kmers_a.intersection(&kmers_b).count();
    let union = kmers_a.union(&kmers_b).count();
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

// =========================================================================
// 4. MAFFT MULTIPLE SEQUENCE ALIGNMENT (pure Rust, no external MAFFT needed)
//    Uses the `mafft` crate (rust-MAFFT workspace) for self-contained MSA.
// =========================================================================

/// Python: mafft_align(sequences: list[tuple[str, str]], mode: str) -> list[tuple[str, str]]
///         Aligns sequences using the pure-Rust MAFFT engine.
///         sequences = [(name, sequence), ...]
///         mode = "fft-ns-2" (fast, default) | "fft-ns-i" | "nw" | "auto"
///         Returns [(name, aligned_sequence), ...]
#[pyfunction]
fn mafft_align(
    py: Python<'_>,
    sequences: Vec<(String, String)>,
    mode: &str,
) -> PyResult<Vec<(String, String)>> {
    use mafft::{MafftEngine, AlignmentMode, SequenceSet, Sequence, SeqType};

    // Detect sequence type (protein vs DNA)
    let all_chars: Vec<u8> = sequences.iter()
        .flat_map(|(_, s)| s.bytes())
        .collect();
    let dna_fraction = all_chars.iter()
        .filter(|&&b| matches!(b, b'A' | b'C' | b'G' | b'T' | b'U' | b'N' | b'a' | b'c' | b'g' | b't' | b'u' | b'n'))
        .count() as f64 / all_chars.len().max(1) as f64;
    let seq_type = if dna_fraction > 0.7 { SeqType::Dna } else { SeqType::Protein };

    // Build SequenceSet
    let mut set = SequenceSet::new(seq_type);
    for (name, seq) in &sequences {
        set.sequences.push(Sequence {
            name: name.clone(),
            data: seq.as_bytes().to_vec(),
        });
    }

    // Select alignment mode
    let alignment_mode = match mode {
        "fft-ns-i" | "fftnsi" => AlignmentMode::FftNsi { iterations: 2 },
        "g-ins-i" | "ginsi" => AlignmentMode::GInsi { iterations: 2 },
        "l-ins-i" | "linsi" => AlignmentMode::LInsi { iterations: 2 },
        "e-ins-i" | "einsi" => AlignmentMode::EInsi { iterations: 2 },
        "auto" | "fft-ns-2" | "fftns2" | _ => AlignmentMode::FftNs2,
    };

    let engine = MafftEngine::new(alignment_mode);

    // Run alignment (release GIL during computation)
    let msa = py.allow_threads(|| engine.align(&set));

    // Convert result to Vec<(String, String)>
    let result: Vec<(String, String)> = msa.names.iter()
        .zip(msa.sequences.iter())
        .map(|(name, seq)| {
            (name.clone(), String::from_utf8_lossy(seq).into_owned())
        })
        .collect();

    Ok(result)
}

/// Covariance-model CYK alignment over a single sequence window.
///
/// The model is passed as flat arrays (built once in Python from the parsed
/// CovarianceModel). Returns (start, end, score) with 1-based inclusive
/// coordinates, or None if no alignment beats `min_score`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cm_cyk(
    py: Python<'_>,
    seq: &str,
    types: Vec<u8>,
    cfirst: Vec<i32>,
    cnum: Vec<i32>,
    trans_flat: Vec<f64>,
    trans_off: Vec<usize>,
    emit_flat: Vec<f64>,
    emit_off: Vec<usize>,
    min_score: f64,
) -> PyResult<Option<(usize, usize, f64)>> {
    let bytes = seq.as_bytes().to_vec();
    let empty: Vec<i32> = Vec::new();
    let result = py.allow_threads(|| {
        let model = cm::Cm {
            types: &types,
            cfirst: &cfirst,
            cnum: &cnum,
            trans_flat: &trans_flat,
            trans_off: &trans_off,
            emit_flat: &emit_flat,
            emit_off: &emit_off,
            state_node: &empty,
            node_lcol: &empty,
            node_rcol: &empty,
        };
        cm::cyk_best(&model, &bytes, min_score)
    });
    Ok(result)
}

/// Covariance-model CYK alignment with traceback.
///
/// Returns (start, end, score, alignment) where alignment is a list of
/// (consensus_column, seq_pos) pairs for match emissions, sorted by consensus
/// column. Requires per-state node indices and per-node consensus columns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn cm_cyk_trace(
    py: Python<'_>,
    seq: &str,
    types: Vec<u8>,
    cfirst: Vec<i32>,
    cnum: Vec<i32>,
    trans_flat: Vec<f64>,
    trans_off: Vec<usize>,
    emit_flat: Vec<f64>,
    emit_off: Vec<usize>,
    state_node: Vec<i32>,
    node_lcol: Vec<i32>,
    node_rcol: Vec<i32>,
) -> PyResult<Option<(usize, usize, f64, Vec<(i32, i32)>)>> {
    let bytes = seq.as_bytes().to_vec();
    let result = py.allow_threads(|| {
        let model = cm::Cm {
            types: &types,
            cfirst: &cfirst,
            cnum: &cnum,
            trans_flat: &trans_flat,
            trans_off: &trans_off,
            emit_flat: &emit_flat,
            emit_off: &emit_off,
            state_node: &state_node,
            node_lcol: &node_lcol,
            node_rcol: &node_rcol,
        };
        cm::cyk_trace_best(&model, &bytes)
    });
    Ok(result)
}

// =========================================================================
// Module registration
// =========================================================================

#[pymodule]
fn organelleverse_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kmer_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(erc_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(kmer_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(mafft_align, m)?)?;
    m.add_function(wrap_pyfunction!(cm_cyk, m)?)?;
    m.add_function(wrap_pyfunction!(cm_cyk_trace, m)?)?;
    Ok(())
}
