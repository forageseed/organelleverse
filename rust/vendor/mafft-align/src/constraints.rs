/// Pairwise local alignment constraint builder.
///
/// Computes all-vs-all pairwise local alignments and stores the results
/// as a `LocalHomologyTable`, which is then used to guide progressive
/// alignment in L-INS-i and E-INS-i modes.

use rayon::prelude::*;

use mafft_types::{HomologyRegion, LocalHomologyTable};

use crate::dp::GapModel;
use crate::local::local_align;

/// MAFFT's default `fastathreshold` for protein alignments (`scripts/mafft:97`).
/// Multiplies each `region.importance * eff1[gi] * eff2[gj]` contribution
/// before it lands in the per-cell importance matrix.
pub const FASTATHRESHOLD_DEFAULT: f64 = 2.7;

/// Build the per-cell importance matrix `impmtx[i][j]`, mirroring C's
/// `fillimp` (`mltaln9.c:15560-15675`).
///
/// For each pair of group members `(s1, s2)`, walk every homology region
/// stored in `localhom.get(s1, s2)`. The region's `start1`/`end1` are in
/// the original raw-sequence position space; `move_to_seq_pos` translates
/// them to positions in the (gapped) sequences passed in `g1_seqs[gi]` /
/// `g2_seqs[gj]`. Within `[start, end]` we step both sequences in lockstep:
/// at every column where neither has a gap, add
/// `region.importance * eff1[gi] * eff2[gj] * fastathreshold` to
/// `impmtx[k1][k2]`.
///
/// `g1_seqs[gi].len()` must equal `lgth1` (same for group 2). Position
/// translation handles within-group gaps by counting non-gap residues.
pub fn build_imp_matrix(
    localhom: &LocalHomologyTable,
    group1: &[usize],
    group2: &[usize],
    g1_seqs: &[&[u8]],
    g2_seqs: &[&[u8]],
    eff1: &[f64],
    eff2: &[f64],
    lgth1: usize,
    lgth2: usize,
    fastathreshold: f64,
) -> Vec<Vec<f64>> {
    let mut imp = vec![vec![0.0f64; lgth2]; lgth1];
    let effijx = fastathreshold;

    for (gi, &s1) in group1.iter().enumerate() {
        for (gj, &s2) in group2.iter().enumerate() {
            let regions = localhom.get(s1, s2);
            if regions.is_empty() { continue; }
            let effij = eff1[gi] * eff2[gj] * effijx;
            let seq1 = g1_seqs[gi];
            let seq2 = g2_seqs[gj];

            for region in regions {
                let start1 = match move_to_seq_pos(seq1, region.start1 as usize) {
                    Some(p) => p,
                    None => continue,
                };
                let end1 = if region.start1 == region.end1 {
                    start1
                } else {
                    match move_to_seq_pos(seq1, region.end1 as usize) {
                        Some(p) => p,
                        None => continue,
                    }
                };
                let start2 = match move_to_seq_pos(seq2, region.start2 as usize) {
                    Some(p) => p,
                    None => continue,
                };
                let end2 = if region.start2 == region.end2 {
                    start2
                } else {
                    match move_to_seq_pos(seq2, region.end2 as usize) {
                        Some(p) => p,
                        None => continue,
                    }
                };

                let mut k1 = start1;
                let mut k2 = start2;
                while k1 < seq1.len() && k2 < seq2.len() {
                    let c1 = seq1[k1];
                    let c2 = seq2[k2];
                    if c1 != b'-' && c2 != b'-' {
                        if k1 < lgth1 && k2 < lgth2 {
                            // C `mltaln9.c:15665`:
                            //   impmtx[k1][k2] += tmpptr->importance * effij;
                            // gcc -O3 fuses into a single `fmadd` (single
                            // rounding). Plain Rust `+=` is 2 roundings, and
                            // the resulting per-cell ULP drift accumulates
                            // over the diagonal sum into a multi-unit
                            // impmatch drift that flips accept/reject
                            // decisions late in iterative refinement
                            // (BB30028 L-INS-i iter=3 fingerprint).
                            imp[k1][k2] = region.importance.mul_add(effij, imp[k1][k2]);
                        }
                        k1 += 1;
                        k2 += 1;
                    } else if c1 != b'-' && c2 == b'-' {
                        k2 += 1;
                    } else if c1 == b'-' && c2 != b'-' {
                        k1 += 1;
                    } else {
                        k1 += 1;
                        k2 += 1;
                    }
                    if k1 > end1 || k2 > end2 { break; }
                }
            }
        }
    }
    if let Ok(f) = std::env::var("RS_IMPMTX_DUMP") {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALL_NO: AtomicUsize = AtomicUsize::new(0);
        let n = CALL_NO.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
            let mut sum = 0.0f64;
            let mut nonzero = 0usize;
            for row in &imp {
                for &v in row {
                    if v != 0.0 { sum += v; nonzero += 1; }
                }
            }
            let d = lgth1.min(lgth2);
            let mut diag_sum = 0.0f64;
            for k in 0..d { diag_sum += imp[k][k]; }
            let imp00 = if lgth1 > 0 && lgth2 > 0 { imp[0][0] } else { 0.0 };
            let _ = writeln!(fp, "call={} lgth1={} lgth2={} nonzero={} sum={:.18e} diag_sum={:.18e} imp00={:.18e}",
                n, lgth1, lgth2, nonzero, sum, diag_sum, imp00);
            // Extended: dump eff + per-pair impmtx[0][0] contribution
            // for the specific BB11005 step-5 shape gate.
            if lgth1 == 355 && lgth2 == 367 {
                let _ = writeln!(fp, "  effijx={:.18e}", effijx);
                let _ = write!(fp, "  eff1=");
                for v in eff1 { let _ = write!(fp, "{:.18e},", v); }
                let _ = writeln!(fp);
                let _ = write!(fp, "  eff2=");
                for v in eff2 { let _ = write!(fp, "{:.18e},", v); }
                let _ = writeln!(fp);
                // Per-pair impmtx[0][0] contribution decomposition.
                for (gi, &s1) in group1.iter().enumerate() {
                    for (gj, &s2) in group2.iter().enumerate() {
                        let regions = localhom.get(s1, s2);
                        let effij = eff1[gi] * eff2[gj] * effijx;
                        let mut hits_at_00 = 0usize;
                        let mut sum_imp_at_00 = 0.0f64;
                        for region in regions {
                            // Mirror the walk; just check if (0,0) gets a hit
                            let seq1 = g1_seqs[gi];
                            let seq2 = g2_seqs[gj];
                            let start1 = move_to_seq_pos(seq1, region.start1 as usize);
                            let start2 = move_to_seq_pos(seq2, region.start2 as usize);
                            if let (Some(s1p), Some(s2p)) = (start1, start2) {
                                if s1p == 0 && s2p == 0 {
                                    hits_at_00 += 1;
                                    sum_imp_at_00 += region.importance;
                                }
                            }
                        }
                        let _ = writeln!(fp,
                            "  pair[{},{}] gi={} gj={} effij={:.18e} regions={} hits_at(0,0)={} sum_imp_at(0,0)={:.18e}  contrib_to_imp00={:.18e}",
                            s1, s2, gi, gj, effij, regions.len(), hits_at_00, sum_imp_at_00, sum_imp_at_00 * effij);
                        // Per-region full dump for these specific pairs.
                        for (rid, region) in regions.iter().enumerate() {
                            let _ = writeln!(fp,
                                "    R_REG pair=[{},{}] r={} s1={} e1={} s2={} e2={} opt={:.18e} overlap={} importance={:.18e}",
                                s1, s2, rid, region.start1, region.end1, region.start2, region.end2, region.opt, region.overlapaa, region.importance);
                        }
                    }
                }
            }
        }
    }
    imp
}

/// Extract local-homology regions from a pair of pre-aligned sequences,
/// mirroring C `putlocalhom2` (`io.c:723`). The two slices must be the
/// same length (gap-aligned). Whenever a gap appears in either column
/// the current match region is closed; a residue-residue column starts
/// a new one.
///
/// `offset1`/`offset2` are the starting raw-residue positions (matching
/// C's `off1`/`off2`); the returned `start1`/`end1`/`start2`/`end2`
/// values are raw-residue indices on top of those offsets.
///
/// `korh` is stamped on every region (`b'h'` for pairwise alignment,
/// `b'k'` for `--seed` constraints).
///
/// The function matches the non-`divpairscore` branch of C
/// (`io.c:855-866`) + `tbfast.c:2202` rescale: all regions share a
/// combined `opt = isumscore / sumoverlap` and `overlapaa = sumoverlap`.
pub fn extract_putlocalhom2_regions(
    al1: &[u8],
    al2: &[u8],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    offset1: i32,
    offset2: i32,
    korh: u8,
) -> Vec<HomologyRegion> {
    let len = al1.len().min(al2.len());
    let n_alpha = matrix.len();
    let mut regions: Vec<HomologyRegion> = Vec::new();
    let mut isumscore: f64 = 0.0;
    let mut sumoverlap: i32 = 0;
    let mut pos1 = offset1;
    let mut pos2 = offset2;
    let mut st = false;
    let mut start1 = 0i32;
    let mut start2 = 0i32;
    let mut iscore: f64 = 0.0;
    for k in 0..len {
        let c1 = al1[k];
        let c2 = al2[k];
        let g1 = c1 == b'-';
        let g2 = c2 == b'-';
        if st && (g1 || g2) {
            let end1 = pos1 - 1;
            let end2 = pos2 - 1;
            regions.push(HomologyRegion {
                start1, end1, start2, end2,
                opt: 0.0,
                overlapaa: end2 - start2 + 1,
                korh,
                ..Default::default()
            });
            isumscore += iscore;
            sumoverlap += end2 - start2 + 1;
            iscore = 0.0;
            st = false;
        } else if !g1 && !g2 {
            if !st {
                start1 = pos1;
                start2 = pos2;
                st = true;
            }
            let i1 = amino_map[c1 as usize] as usize;
            let i2 = amino_map[c2 as usize] as usize;
            if i1 < n_alpha && i2 < n_alpha {
                iscore += matrix[i1][i2];
            }
        }
        if !g1 { pos1 += 1; }
        if !g2 { pos2 += 1; }
    }
    if st {
        let end1 = pos1 - 1;
        let end2 = pos2 - 1;
        regions.push(HomologyRegion {
            start1, end1, start2, end2,
            opt: 0.0,
            overlapaa: end2 - start2 + 1,
            korh,
            ..Default::default()
        });
        isumscore += iscore;
        sumoverlap += end2 - start2 + 1;
    }

    // C MAFFT serializes opt through a `%7.5f` hat3 text file in
    // pairlocalalign.c:3101:
    //   fprintf(hat3p, "%d %d %d %7.5f ...", overlapaa, opt, ...)
    // where `opt = isumscore * 5.8 / (600 * sumoverlap)` (io.c:861).
    // dvtditr then reads opt and rescales it back via
    // io.c:4451: `tmpptr->opt = (opt + 0.00) / 5.8 * 600`.
    // The 5-decimal text round-trip introduces ~1e-5 precision loss
    // relative to the pure-FP `isumscore / sumoverlap` formula.
    // Mirror that loss to stay byte-identical with C — without this,
    // BB30028 / BB50001 L-INS-i diverge by 4 / 244 lines because the
    // ~5e-6-per-region drift accumulates over impmatch_diagonal sums
    // and flips one accept/reject decision late in refinement.
    let opt = if sumoverlap > 0 {
        // Mirror C's hat3 round-trip:
        //   tbfast.c:2218: opt = opt / 5.8 * 600        (in-memory rescale)
        //   tbfast.c:2265: fprintf %7.5f of opt/600*5.8 (write hat3)
        //   io.c:4451:     opt = parsed / 5.8 * 600      (read hat3)
        // The 5-decimal rounding uses C's printf which is **round-half-
        // to-even** on macOS/glibc. Rust's `f64::round()` is round-half-
        // away-from-zero — they disagree at exact half-points like the
        // BB20004 (31,36) case where opt_pre*1e5 = 197620.5 exactly.
        // Using `format!("{:.5}", v).parse()` reproduces printf's banker's
        // rounding.
        let opt_pre = isumscore * 5.8 / (600.0 * sumoverlap as f64);
        let opt_rescaled = opt_pre / 5.8 * 600.0;
        let hat3_value = opt_rescaled / 600.0 * 5.8;
        let opt_file: f64 = format!("{:.5}", hat3_value).parse().unwrap();
        opt_file / 5.8 * 600.0
    } else { 0.0 };
    let provisional_importance =
        if sumoverlap > 0 { opt / sumoverlap as f64 } else { 0.0 };
    for r in regions.iter_mut() {
        r.opt = opt;
        r.overlapaa = sumoverlap;
        r.importance = provisional_importance;
    }
    regions
}

/// One pre-aligned seed file: gapped sequences plus their indices in the
/// final (seeds + user input) sequence list. All pairs within the group
/// generate `korh = 'k'` homology regions.
pub struct SeedGroup<'a> {
    /// Aligned sequences (gaps preserved) from this seed file.
    pub aligned: Vec<&'a [u8]>,
    /// Indices into the combined sequence list (seeds + user input)
    /// where these seed sequences live.
    pub global_indices: Vec<usize>,
}

/// Build a `LocalHomologyTable` of dimension `total_nseq` from one or
/// more pre-aligned seed groups. Mirrors C `multi2hat3s` (`multi2hat3s.c`)
/// + `tbfast.c:2202` rescale:
///
/// 1. For every (i, j) pair WITHIN a seed group, run
///    `extract_putlocalhom2_regions` on the two gapped strings to obtain
///    chained match regions.
/// 2. Multiply each region's `opt` (already in `tbfast.c:2202`-post-scale
///    form, i.e. `isumscore / sumoverlap`) by `tsuyosa = user_nseq² *
///    TSUYOSAFACTOR (100)`. This boosts seed importance over regular
///    pairwise homology by a factor proportional to N².
/// 3. Recompute the provisional importance (`opt / sumoverlap`) so the
///    boosted `opt` is reflected before `recompute_importance` runs.
/// 4. Mirror the entry symmetrically: `(i, j)` AND `(j, i)` (`j, i` with
///    `start1/start2` swapped) — matching how `build_local_homology_table`
///    already populates both directions.
///
/// Different seed groups do not share homology entries (the C script
/// invokes `multi2hat3s` separately per seed file, with disjoint
/// `seedoffset` ranges).
pub fn build_seed_homology_table(
    seed_groups: &[SeedGroup<'_>],
    total_nseq: usize,
    user_nseq: usize,
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
) -> LocalHomologyTable {
    /// Matches C `multi2hat3s.c::TSUYOSAFACTOR`.
    const TSUYOSAFACTOR: f64 = 100.0;
    let tsuyosa = (user_nseq as f64) * (user_nseq as f64) * TSUYOSAFACTOR;
    let mut table = LocalHomologyTable::new(total_nseq);

    for group in seed_groups {
        let n = group.aligned.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let gi = group.global_indices[i];
                let gj = group.global_indices[j];
                let regions = extract_putlocalhom2_regions(
                    group.aligned[i], group.aligned[j],
                    matrix, amino_map, 0, 0, b'k',
                );
                if regions.is_empty() { continue; }
                let overlapaa = regions[0].overlapaa;
                let boosted_importance = if overlapaa > 0 {
                    (regions[0].opt * tsuyosa) / overlapaa as f64
                } else { 0.0 };
                for r in &regions {
                    let mut fwd = r.clone();
                    fwd.opt = r.opt * tsuyosa;
                    fwd.importance = boosted_importance;
                    let rev = HomologyRegion {
                        start1: fwd.start2,
                        end1: fwd.end2,
                        start2: fwd.start1,
                        end2: fwd.end1,
                        ..fwd.clone()
                    };
                    table.push(gi, gj, fwd);
                    table.push(gj, gi, rev);
                }
            }
        }
    }
    table
}

/// Parse a hat3 seed table (matches C MAFFT's `--seedtable` input format,
/// which is the same format `multi2hat3s.c:214` writes for `--seed`).
///
/// Each non-empty line:
///
/// ```text
/// i j overlapaa opt start1 end1 start2 end2 k
/// ```
///
/// where `i`, `j` are 0-based sequence indices into the alignment input,
/// `start*` / `end*` are inclusive 0-based residue positions, `opt` is
/// the `tsuyosa`-boosted-and-`*5.8/600`-normalized score that C's
/// `putlocalhom2` (`io.c:861`) writes, and `k` is the `korh`
/// classification byte. The C reader (`readlocalhomtable2_half`) populates
/// only one triangle; we mirror each entry to `(j, i)` with
/// `start1`/`start2` swapped so downstream `build_imp_matrix` /
/// `recompute_importance` find entries in both directions.
///
/// `opt` is stored as-written by `multi2hat3s.c:214`
/// (`isumscore * 5.8 / (600 * sumoverlap) * tsuyosa`); C loads these
/// directly into its localhomtable via `readlocalhomtable2_half` without
/// any rescale, so we do the same. Note this is ~100× smaller than the
/// magnitude `build_seed_homology_table` produces (the in-memory `--seed`
/// path skips both C's `5.8/600` write-side and C's pairwise `600/5.8`
/// read-side, while the seed-load path applies neither). Both magnitudes
/// dominate the per-cell impmatrix contributions so the final alignment
/// is unchanged; what matters is internal self-consistency with C.
///
/// Blank lines and lines starting with `#` are skipped (matching the
/// `grep -v "^$"` pre-filter in `scripts/mafft:1149`). Any malformed line
/// returns an `Err` with a 1-based line number for diagnostics.
pub fn parse_hat3_seed(
    text: &str,
    nseq: usize,
) -> Result<LocalHomologyTable, String> {
    let mut table = LocalHomologyTable::new(nseq);
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 9 {
            return Err(format!(
                "hat3 line {}: expected 9 fields, got {}", lineno + 1, parts.len()));
        }
        let parse_usize = |s: &str, field: &str| -> Result<usize, String> {
            s.parse::<usize>().map_err(|e| format!(
                "hat3 line {}: parse `{}` ({}): {}", lineno + 1, field, s, e))
        };
        let parse_i32 = |s: &str, field: &str| -> Result<i32, String> {
            s.parse::<i32>().map_err(|e| format!(
                "hat3 line {}: parse `{}` ({}): {}", lineno + 1, field, s, e))
        };
        let parse_f64 = |s: &str, field: &str| -> Result<f64, String> {
            s.parse::<f64>().map_err(|e| format!(
                "hat3 line {}: parse `{}` ({}): {}", lineno + 1, field, s, e))
        };
        let i = parse_usize(parts[0], "i")?;
        let j = parse_usize(parts[1], "j")?;
        let overlapaa = parse_i32(parts[2], "overlapaa")?;
        let opt = parse_f64(parts[3], "opt")?;
        let start1 = parse_i32(parts[4], "start1")?;
        let end1 = parse_i32(parts[5], "end1")?;
        let start2 = parse_i32(parts[6], "start2")?;
        let end2 = parse_i32(parts[7], "end2")?;
        let korh = parts[8].as_bytes().first().copied().unwrap_or(b'k');
        if i >= nseq || j >= nseq {
            return Err(format!(
                "hat3 line {}: index ({}, {}) out of range for nseq={}",
                lineno + 1, i, j, nseq));
        }
        if i == j { continue; }
        let importance = if overlapaa > 0 { opt / overlapaa as f64 } else { 0.0 };
        let fwd = HomologyRegion {
            start1, end1, start2, end2,
            opt,
            overlapaa,
            importance,
            korh,
            ..Default::default()
        };
        let rev = HomologyRegion {
            start1: start2,
            end1: end2,
            start2: start1,
            end2: end1,
            ..fwd.clone()
        };
        table.push(i, j, fwd);
        table.push(j, i, rev);
    }
    Ok(table)
}

/// Merge `extra` entries into `into`. Both tables must have the same
/// `nseq`. Used to fold a seed-derived homology table into the pairwise
/// homology table built by L-INS-i / G-INS-i / E-INS-i.
pub fn merge_homology_tables(into: &mut LocalHomologyTable, extra: &LocalHomologyTable) {
    if into.nseq != extra.nseq { return; }
    for i in 0..into.nseq {
        for j in 0..into.nseq {
            if i == j { continue; }
            for r in extra.get(i, j) {
                into.push(i, j, r.clone());
            }
        }
    }
}

/// Recompute homology-region `importance` values using C's
/// position-vote algorithm in `calcimportance` (`mltaln9.c:11984`).
///
/// Replaces each region's provisional importance (typically `opt /
/// overlapaa` from `build_local_homology_table`) with a value that
/// rewards regions backed by many "voting" sequences:
///
/// 1. Normalize `ieff[j] = eff[j] / sum(eff[non-empty])`.
/// 2. For each sequence `i`, build a per-position support array
///    `support[pos] = sum_j ieff[j]` over every region in
///    `localhom.get(i, j)` that covers position `pos` (raw residue
///    index in `sequences[i]`).
/// 3. For each region in `localhom.get(i, j)`, set
///    `region.importance = mean(support[start1..=end1]) * region.opt`.
/// 4. Symmetrize: for each pair `(i, j)`, average the importance of
///    the matching regions in `localhom.get(i, j)` and `localhom.get(j, i)`.
///
/// `eff` is the global tree-weight vector (typically from
/// `mafft-tree::sequence_weights`) — match the value C's `tbfast` passes
/// to `calcimportance` after building the tree.
pub fn recompute_importance(
    localhom: &mut LocalHomologyTable,
    sequences: &[&[u8]],
    eff: &[f64],
) {
    let nseq = sequences.len();
    if nseq < 2 || localhom.nseq != nseq { return; }

    let nogaplen: Vec<usize> = sequences.iter()
        .map(|s| s.iter().filter(|&&c| c != b'-').count()).collect();
    // C `mltaln9.c:11806-11814` totaleff computation mirrors:
    //   totaleff = 0.0;
    //   for(i=0; i<nseq; i++) {
    //     ieff[i] = (nogaplen[i] > 0) ? eff[i] : 0.0;
    //     totaleff += ieff[i];
    //   }
    // Sequential `+=` to match C's accumulation order exactly.
    let mut totaleff = 0.0f64;
    for i in 0..nseq {
        if nogaplen[i] > 0 { totaleff += eff[i]; }
    }
    if totaleff <= 0.0 { return; }
    let ieff: Vec<f64> = (0..nseq).map(|i| {
        if nogaplen[i] > 0 { eff[i] / totaleff } else { 0.0 }
    }).collect();
    let nlenmax = nogaplen.iter().copied().max().unwrap_or(0);
    if nlenmax == 0 { return; }

    // Pass 1: per-i position-vote → set importance = mean(support over region) * opt
    let mut support = vec![0.0f64; nlenmax];
    for i in 0..nseq {
        for v in support.iter_mut() { *v = 0.0; }
        for j in 0..nseq {
            if i == j { continue; }
            for region in localhom.get(i, j) {
                let s = (region.start1 as usize).min(nlenmax);
                let e = (region.end1 as usize).min(nlenmax.saturating_sub(1));
                for pos in s..=e {
                    if pos < nlenmax { support[pos] += ieff[j]; }
                }
            }
        }
        if let Ok(f) = std::env::var("RS_SUPPORT_DUMP") {
            use std::io::Write;
            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
                let _ = writeln!(fp, "R_SUPPORT i={} ieff={:?} support[0..5]={:?}", i, &ieff[..nseq.min(ieff.len())], &support[..5.min(support.len())]);
            }
        }
        for j in 0..nseq {
            if i == j { continue; }
            // Note: C iterates `for tmpptr = localhom[i]+j; tmpptr; tmpptr=tmpptr->next`
            // — a linked list. We have a Vec; iterate by index so we can mutate.
            let regions_count = localhom.get(i, j).len();
            for r in 0..regions_count {
                let (s, e) = {
                    let region = &localhom.get(i, j)[r];
                    (
                        (region.start1 as usize).min(nlenmax.saturating_sub(1)),
                        (region.end1 as usize).min(nlenmax.saturating_sub(1)),
                    )
                };
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for pos in s..=e {
                    sum += support[pos];
                    count += 1;
                }
                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                let region = &mut localhom.get_mut(i, j)[r];
                region.importance = mean * region.opt;
            }
        }
    }

    // Pass 2: average importance between (i,j) and (j,i) directions to
    // mirror C's `calcimportance_half` final averaging loop
    // (`mltaln9.c:11928-11950`):
    //   imp = 0.5 * (tmpptr1->importance + tmpptr1->rimportance);
    //   tmpptr1->importance = tmpptr1->rimportance = imp;
    // C's half-matrix struct stores both `importance` and `rimportance`
    // for the same pair; the averaging makes them equal. In our full-matrix
    // layout, (i,j).importance corresponds to C's `importance` and (j,i)
    // .importance corresponds to C's `rimportance`. Averaging both back to
    // the same value matches C's final state.
    for i in 0..nseq.saturating_sub(1) {
        for j in (i + 1)..nseq {
            let n_ij = localhom.get(i, j).len();
            let n_ji = localhom.get(j, i).len();
            let n = n_ij.min(n_ji);
            for r in 0..n {
                let avg = 0.5 * (
                    localhom.get(i, j)[r].importance
                  + localhom.get(j, i)[r].importance
                );
                localhom.get_mut(i, j)[r].importance = avg;
                localhom.get_mut(j, i)[r].importance = avg;
            }
        }
    }
}

/// Translate a raw-sequence residue index (`raw_pos`) into a position in a
/// (possibly gapped) sequence. Mirrors C's `movereg` (`mltaln9.c:15467`):
/// walk forward, count non-gap characters, return the index of the
/// `raw_pos`-th residue. If `raw_pos` exceeds the sequence's residue count
/// (e.g. an exclusive end-marker past the last residue), returns the index
/// just after the last residue, so callers using `<=` end checks still
/// terminate cleanly.
fn move_to_seq_pos(seq: &[u8], raw_pos: usize) -> Option<usize> {
    let target = raw_pos as i64;
    let mut count: i64 = -1;
    let mut last_residue_idx = 0usize;
    let mut found_any = false;
    for (idx, &c) in seq.iter().enumerate() {
        if c != b'-' {
            count += 1;
            last_residue_idx = idx;
            found_any = true;
        }
        if count == target { return Some(idx); }
    }
    if found_any && raw_pos > count as usize {
        Some(last_residue_idx + 1)
    } else {
        None
    }
}

/// Result of one pairwise alignment for parallel collection.
struct PairResult {
    i: usize,
    j: usize,
    distance: f64,
    regions: Vec<HomologyRegion>,
    #[allow(dead_code)]
    _debug_pscore: f64,
    #[allow(dead_code)]
    _debug_bunbo: f64,
}

/// Which pairwise aligner to drive `build_homology_table` with.
///
/// L-INS-i uses `Local` (Smith–Waterman, mirroring C's `pairlocalalign -L`
/// → `L__align11`). G-INS-i uses `Global` (Needleman–Wunsch, mirroring
/// `pairlocalalign -A` → `G__align11`). E-INS-i uses
/// `GeneralizedAffine` (Smith-Waterman with an extra "skip" gap state,
/// mirroring `pairlocalalign -N` → `genL__align11`). The chaining/`opt`
/// computation downstream is identical — only the alignment differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairAligner {
    Local,
    Global,
    GeneralizedAffine,
}


/// Build a local homology table from all-vs-all pairwise alignments.
///
/// Defaults to local Smith-Waterman. See `build_homology_table` for the
/// unified entry point that chooses between local and global.
pub fn build_local_homology_table(
    sequences: &[&[u8]],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap: &GapModel,
    score_offset: f64,
) -> (LocalHomologyTable, Vec<Vec<f64>>) {
    build_homology_table(
        sequences, matrix, amino_map, gap, score_offset,
        PairAligner::Local, 0.0,
    )
}

/// Build a homology table using all-vs-all pairwise alignment of the
/// chosen kind. Used by L-INS-i (`PairAligner::Local`), G-INS-i
/// (`PairAligner::Global`), and E-INS-i (`PairAligner::GeneralizedAffine`).
///
/// `op_penalty` is the generalized-affine "skip" open penalty (C's
/// `penalty_OP`), used only when `aligner == GeneralizedAffine`.
pub fn build_homology_table(
    sequences: &[&[u8]],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap: &GapModel,
    score_offset: f64,
    aligner: PairAligner,
    op_penalty: f64,
) -> (LocalHomologyTable, Vec<Vec<f64>>) {
    build_homology_table_with_unalign(
        sequences, matrix, amino_map, gap, score_offset, aligner, op_penalty, 0.0,
    )
}

/// Like `build_homology_table` but with per-pair dynamic-matrix
/// re-alignment when `unalign_level > 0`. Mirrors C's
/// `pairlocalalign.c:2196-2228` (case 'A') and `:2237-2253` (case 'l'):
/// after the initial pairwise alignment, compute
/// `dist = score2dist(score, selfscore[i], selfscore[j])`. If
/// `dist2offset(dist) < 0` (= `0.5*dist - unalign_level < 0`), build a
/// dynamic substitution matrix with offset `(0.5*dist - unalign_level) *
/// 600` and re-run the pairwise alignment. The new alignment is used
/// for the local-homology constraints; the *original* score is kept for
/// the distance matrix (matching C, which never assigns the second
/// `G__align11`'s return value back to `pscore`).
pub fn build_homology_table_with_unalign(
    sequences: &[&[u8]],
    matrix: &[Vec<f64>],
    amino_map: &[u8; 256],
    gap: &GapModel,
    score_offset: f64,
    aligner: PairAligner,
    op_penalty: f64,
    unalign_level: f64,
) -> (LocalHomologyTable, Vec<Vec<f64>>) {
    let nseq = sequences.len();

    // C's `pairlocalalign.c:2590-2596`: selfscore[i] = sum of diagonal
    // substitution-matrix entries for each residue in the sequence,
    // using the same offset-shifted matrix that pairwise alignment uses.
    // Used by `score2dist(pscore, selfscore[i], selfscore[j])`
    // (line 1931) to convert alignment scores to tree-input distances:
    //   bunbo = min(selfscore[i], selfscore[j])
    //   dist = (1 - pscore / bunbo) * 2  (clamped to [0, 2])
    let n_alpha = matrix.len();
    let selfscore: Vec<f64> = sequences.iter().map(|s| {
        let mut sum = 0.0f64;
        for &c in *s {
            let i = amino_map[c as usize] as usize;
            if i < n_alpha {
                sum += matrix[i][i];
            }
        }
        sum
    }).collect();

    // Generate all (i, j) pairs with i < j
    let pairs: Vec<(usize, usize)> = (0..nseq)
        .flat_map(|i| ((i + 1)..nseq).map(move |j| (i, j)))
        .collect();

    // Compute all pairwise alignments in parallel
    let results: Vec<PairResult> = pairs
        .par_iter()
        .map(|&(i, j)| {
            // Branch at the alignment call to keep the rest of the
            // chaining logic shared. For Local, we have a true offset
            // into both sequences. For Global, both offsets are 0.
            let run_align = |mat: &[Vec<f64>]| -> (crate::dp::Alignment, usize, usize) {
                match aligner {
                    PairAligner::Local => {
                        let r = local_align(
                            sequences[i], sequences[j],
                            mat, amino_map, gap, score_offset,
                        );
                        (r.alignment, r.offset1, r.offset2)
                    }
                    PairAligner::Global => {
                        // Match C's `pairlocalalign -A` (`pairlocalalign.c:2196`)
                        // which calls `G__align11` with `outgap` controlling
                        // terminal-gap treatment. The script for G-INS-i
                        // doesn't pass `-O`, so `outgap` defaults to 1 →
                        // both head and tail gaps are penalized.
                        let r = crate::global::global_align(
                            sequences[i], sequences[j],
                            mat, amino_map, gap,
                            true, true,
                        );
                        if let Ok(p) = std::env::var("RS_PAIR_TRACE") {
                            // Serialize the trace write via a global mutex so
                            // parallel pairs don't interleave their lines.
                            use std::io::Write;
                            use std::sync::{Mutex, OnceLock};
                            static TRACE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
                            let lk = TRACE_LOCK.get_or_init(|| Mutex::new(()));
                            let _guard = lk.lock().unwrap();
                            if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                                let s1 = std::str::from_utf8(&r.seq1).unwrap_or("?");
                                let s2 = std::str::from_utf8(&r.seq2).unwrap_or("?");
                                let _ = writeln!(
                                    fp,
                                    "R_PAIR i={} j={} n={} m={} gap.shift={:?}\nR_PAIR_SCORE i={} j={} score={:.17e} aln1={} aln2={}",
                                    i, j, sequences[i].len(), sequences[j].len(), gap.shift,
                                    i, j, r.score, s1, s2,
                                );
                            }
                        }
                        (r, 0, 0)
                    }
                    PairAligner::GeneralizedAffine => {
                        // C's `pairlocalalign -N` (`pairlocalalign.c:2233`) →
                        // `genL__align11`: max-so-far Smith-Waterman with an
                        // extra "skip" gap state (penalty_OP, no extension).
                        use crate::genaffine::{genaffine_local_align, GenAffineGapModel};
                        let gen_gap = GenAffineGapModel {
                            affine: gap.clone(),
                            open_generalized: op_penalty,
                        };
                        let r = genaffine_local_align(
                            sequences[i], sequences[j],
                            mat, amino_map, &gen_gap, score_offset,
                        );
                        (r.alignment, r.offset1, r.offset2)
                    }
                }
            };
            let (mut alignment, mut offset1, mut offset2) = run_align(matrix);

            // C `pairlocalalign.c:2197`: when `thereisx`, pscore for distance
            // is recomputed via `G__align11_noalign` on X-stripped sequences,
            // BEFORE specificityconsideration / dynmtx construction. Rust
            // previously did this strip AFTER the dynmtx re-run, causing the
            // dynmtx to be built from the with-X (wrong) dist when sequences
            // contained X residues. That mismatch propagated through the
            // second G__align11 → wrong LH regions / opt / impmtx values
            // (BB11005 step 5 impmtx[0][0] off by 12).
            let pscore_for_dist: f64 = if matches!(aligner, PairAligner::Local | PairAligner::Global) {
                let has_x_i = sequences[i].iter().any(|&c| c == b'X' || c == b'x');
                let has_x_j = sequences[j].iter().any(|&c| c == b'X' || c == b'x');
                if has_x_i || has_x_j {
                    let strip = |s: &[u8]| -> Vec<u8> {
                        s.iter().copied().filter(|&c| c != b'X' && c != b'x').collect()
                    };
                    let s1 = strip(sequences[i]);
                    let s2 = strip(sequences[j]);
                    match aligner {
                        PairAligner::Local => {
                            local_align(&s1, &s2, matrix, amino_map, gap, score_offset).alignment.score
                        }
                        PairAligner::Global => {
                            crate::global::global_align(&s1, &s2, matrix, amino_map, gap, true, true).score
                        }
                        _ => alignment.score,
                    }
                } else {
                    alignment.score
                }
            } else {
                alignment.score
            };
            // Apply the X-stripped score to alignment.score UNCONDITIONALLY
            // (i.e. regardless of unalign_level). This matches C's flow where
            // `pscore = G__align11_noalign(distseq, ...)` overwrites pscore
            // BEFORE both the specificityconsideration block and the final
            // `pscore = score2dist(pscore, ...)` (`pairlocalalign.c:2197,2210,2384`).
            // For non-allowshift L/G-INS-i with X-containing inputs, this is
            // the score used for the distance matrix → the guide tree → the
            // entire progressive alignment.
            alignment.score = pscore_for_dist;

            // Per-pair dynamic re-alignment (C `pairlocalalign.c:2199-2215`):
            // when `specificityconsideration > 0` and the initial alignment's
            // distance falls under `2*unalign_level`, re-run the pairwise DP
            // with a substitution matrix scaled by `(0.5*dist - unalign_level)
            // * 600`. The original `alignment.score` is kept for the distance
            // matrix; only the alignment trace is replaced (used for the
            // local-homology region extraction below).
            if unalign_level > 0.0 && pscore_for_dist > 0.0 {
                let bunbo = selfscore[i].min(selfscore[j]);
                let dist_for_offset = if bunbo == 0.0 {
                    2.0
                } else if bunbo < pscore_for_dist {
                    0.0
                } else {
                    (1.0 - pscore_for_dist / bunbo) * 2.0
                };
                let off = 0.5 * dist_for_offset - unalign_level;
                if off < 0.0 {
                    // C `mltaln9.c::makedynamicmtx` adds `offset * 600` to
                    // every cell EXCEPT where amino[i] or amino[j] is '-'
                    // (mltaln9.c:15197-15203). For protein this is index 24,
                    // for DNA index 24. We use `amino_map[b'-']` to look it
                    // up dynamically. Cells in the '-' row/col stay at their
                    // un-delta values. This matters because the C profile DP
                    // / global DP reads matrix[gap_idx][...] on certain code
                    // paths (e.g. via amino_dynamicmtx char-indexing in C).
                    let gap_idx = amino_map[b'-' as usize] as usize;
                    // C `mltaln9.c::makedynamicmtx` computes
                    //     out[i][j] = in[i][j] + offset * 600
                    // per cell, which clang at -O3 with the default
                    // FP_CONTRACT=on lowers to a single FMA
                    // `fmadd(offset, 600, in[i][j])`. Precomputing
                    // `delta = off * 600` and doing `v + delta` is two
                    // rounded ops and drifts ~1 ULP per cell — enough to
                    // flip a tied DP cell (BB12003 first-divergent cell
                    // (2, 18) in the warp pairwise DP). Use mul_add to
                    // mirror C's FMA fusion exactly. Same fix shape as
                    // §B.10 (calcW) and the other FP_CONTRACT divergences.
                    let dyn_matrix: Vec<Vec<f64>> = matrix
                        .iter().enumerate()
                        .map(|(i, row)| {
                            row.iter().enumerate()
                                .map(|(j, &v)| {
                                    if i == gap_idx || j == gap_idx {
                                        v
                                    } else {
                                        off.mul_add(600.0, v)
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    // §E.1 forensic: dump dist / off / delta / dyn_matrix
                    // row M (12) to compare with C's makedynamicmtx output.
                    if let Ok(p) = std::env::var("RS_DYNMTX_DUMP") {
                        use std::io::Write;
                        if let Ok(mut fp) = std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                            let _ = writeln!(fp, "R_DYNMTX_INPUTS i={} j={} selfi={:.17e} selfj={:.17e} bunbo={:.17e} pscore={:.17e}",
                                i, j, selfscore[i], selfscore[j], bunbo, alignment.score);
                            // delta is no longer pre-computed (now fused via mul_add per cell);
                            // print off*600 as the reference for diffing.
                            let _ = writeln!(fp, "R_DYNMTX_OUTPUT dist={:.17e} off={:.17e} delta={:.17e}",
                                dist_for_offset, off, off * 600.0);
                            let _ = write!(fp, "R_DYNMTX_R12");
                            for k in 0..dyn_matrix[12].len() { let _ = write!(fp, " {:.17e}", dyn_matrix[12][k]); }
                            let _ = writeln!(fp);
                            let _ = write!(fp, "R_DYNMTX_R0");
                            for k in 0..dyn_matrix[0].len() { let _ = write!(fp, " {:.17e}", dyn_matrix[0][k]); }
                            let _ = writeln!(fp);
                        }
                    }
                    let (re_aln, re_off1, re_off2) = run_align(&dyn_matrix);
                    alignment = re_aln;
                    offset1 = re_off1;
                    offset2 = re_off2;
                    // Restore C's invariant: the distance comes from the
                    // *original* score (`pairlocalalign.c:2204` reads
                    // `pscore` from the first alignment, not the re-aligned
                    // one). `pscore_for_dist` is set above to the (possibly
                    // X-stripped) first-call score.
                    alignment.score = pscore_for_dist;
                }
            }

            // For E-INS-i, C's `pairlocalalign -N -Z`
            // (`scripts/mafft:1946`, `pairlocalalign.c:2225-2229`) overrides
            // the alignment-derived `pscore` with `naivepairscore11`:
            //   pscore = sum over aligned non-gap columns of amino_dis[c1][c2]
            //   (gap columns get `penal = 0.0` from line 2228).
            // The genL__align11 score is used only to derive the alignment;
            // the distance uses the naive sum-of-pairs over that alignment.
            //
            // For L-INS-i / G-INS-i, the alignment score is used directly,
            // EXCEPT when either input has X residues. C strips X via
            // `removex` (`pairlocalalign.c:3277`) then recomputes pscore via
            // `L__align11_noalign` (`pairlocalalign.c:2139`). To match, we
            // recompute the score on X-stripped sequences when needed; the
            // alignment regions still come from the X-containing run.
            let score_for_dist: f64 = if matches!(aligner, PairAligner::GeneralizedAffine) {
                // commongappick + sum amino_dis at matched columns; gaps free.
                let mut s = 0.0f64;
                let n_alpha = matrix.len();
                for k in 0..alignment.seq1.len() {
                    let c1 = alignment.seq1[k];
                    let c2 = alignment.seq2[k];
                    if c1 == b'-' || c2 == b'-' { continue; }
                    let i1 = amino_map[c1 as usize] as usize;
                    let i2 = amino_map[c2 as usize] as usize;
                    if i1 < n_alpha && i2 < n_alpha {
                        s += matrix[i1][i2];
                    }
                }
                s
            } else {
                alignment.score
            };

            // X-override for L/G aligners: now computed upfront as
            // `pscore_for_dist` and threaded through `alignment.score` via
            // the restore. The downstream X-strip block previously here
            // was redundant (computed the same value twice and applied it
            // AFTER dynmtx construction, the source of the BB11005 bug).
            // Kept as a no-op: pscore_for_dist is already in score_for_dist
            // via the alignment.score restore.
            let _ = pscore_for_dist; // anchor the binding visibly

            // C's `score2dist` (`mltaln9.c:4329-4342` / `pairlocalalign.c:1931-1944`):
            //   bunbo = min(selfscore[i], selfscore[j])
            //   dist = bunbo == 0           ? 2.0
            //        : bunbo < pscore       ? 0.0
            //        : (1 - pscore / bunbo) * 2
            // Note: C does NOT clamp the result, so negative pscores produce
            // dist > 2.0 (e.g. BB40037 pair (3,21) at G-INS-i: pscore ≈ -145,
            // bunbo ≈ 60074 → dist ≈ 2.0048). UPGMA reads these as more-distant
            // pairs; capping at 2.0 changes downstream branch lengths. An
            // earlier `if score_for_dist <= 0.0 { return 2.0 }` guard here
            // diverged from C on pairs with poor-quality global alignments.
            let bunbo = selfscore[i].min(selfscore[j]);
            let d = if bunbo == 0.0 {
                2.0
            } else if bunbo < score_for_dist {
                0.0
            } else {
                (1.0 - score_for_dist / bunbo) * 2.0
            };

            // Port of C's `putlocalhom2` (`io.c:723`) via the shared
            // helper. We bypass C's `* 5.8/600` hat3-normalization round-
            // trip and store the post-`tbfast.c:2202`-scale value
            // (`isumscore / sumoverlap`) directly — what
            // `calcimportance`/`fillimp` actually consume.
            let regions = extract_putlocalhom2_regions(
                &alignment.seq1, &alignment.seq2,
                matrix, amino_map, offset1 as i32, offset2 as i32, b'h',
            );

            PairResult { i, j, distance: d, regions, _debug_pscore: score_for_dist, _debug_bunbo: bunbo }
        })
        .collect();

    if let Ok(f) = std::env::var("RS_LINSI_DUMP") {
        use std::io::Write;
        if let Ok(mut fp) = std::fs::File::create(&f) {
            let mut sorted: Vec<&PairResult> = results.iter().collect();
            sorted.sort_by_key(|r| (r.i, r.j));
            for r in &sorted {
                let _ = writeln!(fp, "pair[{},{}] pscore={:.18e} bunbo={:.18e} dist={:.18e}", r.i, r.j, r._debug_pscore, r._debug_bunbo, r.distance);
            }
        }
    }
    if let Ok(f) = std::env::var("RS_LINSI_HAT3") {
        use std::io::Write;
        if let Ok(mut fp) = std::fs::File::create(&f) {
            let mut sorted: Vec<&PairResult> = results.iter().collect();
            sorted.sort_by_key(|r| (r.i, r.j));
            for r in &sorted {
                for region in &r.regions {
                    let _ = writeln!(fp, "{} {} {} {:7.5} {} {} {} {} h",
                        r.i, r.j, region.overlapaa, region.opt,
                        region.start1, region.end1, region.start2, region.end2);
                }
            }
        }
    }

    // Apply results to table and distance matrix (sequential)
    let mut table = LocalHomologyTable::new(nseq);
    let mut dist = vec![vec![0.0f64; nseq]; nseq];

    for r in results {
        dist[r.i][r.j] = r.distance;
        dist[r.j][r.i] = r.distance;

        for region in &r.regions {
            table.push(r.i, r.j, region.clone());
            table.push(r.j, r.i, HomologyRegion {
                start1: region.start2,
                end1: region.end2,
                start2: region.start1,
                end2: region.end1,
                ..region.clone()
            });
        }
    }

    (table, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_setup() -> (Vec<Vec<f64>>, [u8; 256]) {
        let mut mtx = vec![vec![-100.0f64; 5]; 5];
        for i in 0..4 { mtx[i][i] = 100.0; }
        let mut map = [0xFFu8; 256];
        map[b'A' as usize] = 0; map[b'C' as usize] = 1;
        map[b'G' as usize] = 2; map[b'T' as usize] = 3;
        map[b'-' as usize] = 4;
        (mtx, map)
    }

    #[test]
    fn builds_table_for_similar_sequences() {
        let (mtx, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGT"];
        let gap = GapModel::new(-200.0, -10.0);

        let (table, dist) = build_local_homology_table(&seqs, &mtx, &map, &gap, 0.0);

        let regions_01 = table.get(0, 1);
        assert!(!regions_01.is_empty(), "should find homology between identical seqs");
        assert!(dist[0][1] < 1e-6, "identical seqs should have distance ~0, got {}", dist[0][1]);
    }

    #[test]
    fn reciprocal_entries() {
        let (mtx, map) = simple_setup();
        let seqs: Vec<&[u8]> = vec![b"ACGTACGT", b"ACGTACGT"];
        let gap = GapModel::new(-200.0, -10.0);

        let (table, _) = build_local_homology_table(&seqs, &mtx, &map, &gap, 0.0);

        let fwd = table.get(0, 1);
        let rev = table.get(1, 0);
        assert_eq!(fwd.len(), rev.len());

        if !fwd.is_empty() {
            assert_eq!(fwd[0].start1, rev[0].start2);
            assert_eq!(fwd[0].start2, rev[0].start1);
        }
    }

    #[test]
    fn parse_hat3_seed_basic() {
        let text = "0 1 10 1234.500 0 9 0 9 k\n";
        let table = parse_hat3_seed(text, 3).expect("parse ok");
        let fwd = table.get(0, 1);
        assert_eq!(fwd.len(), 1);
        assert_eq!(fwd[0].start1, 0);
        assert_eq!(fwd[0].end1, 9);
        assert_eq!(fwd[0].start2, 0);
        assert_eq!(fwd[0].end2, 9);
        assert_eq!(fwd[0].overlapaa, 10);
        assert_eq!(fwd[0].opt, 1234.5);
        assert_eq!(fwd[0].importance, 123.45);
        assert_eq!(fwd[0].korh, b'k');
        let rev = table.get(1, 0);
        assert_eq!(rev.len(), 1);
        assert_eq!(rev[0].start1, 0);
        assert_eq!(rev[0].end1, 9);
        assert_eq!(rev[0].opt, 1234.5);
    }

    #[test]
    fn parse_hat3_seed_skips_blank_and_comments() {
        let text = "\n# leading comment\n0 1 10 100.000 0 9 0 9 k\n\n# trailing\n";
        let table = parse_hat3_seed(text, 2).expect("parse ok");
        assert_eq!(table.get(0, 1).len(), 1);
        assert_eq!(table.get(1, 0).len(), 1);
    }

    #[test]
    fn parse_hat3_seed_rejects_oob_index() {
        let text = "0 5 10 1.0 0 9 0 9 k\n";
        assert!(parse_hat3_seed(text, 3).is_err());
    }

    #[test]
    fn parse_hat3_seed_rejects_short_line() {
        let text = "0 1 10 1.0 0\n";
        assert!(parse_hat3_seed(text, 3).is_err());
    }

    #[test]
    fn parse_hat3_seed_multiple_regions_same_pair() {
        // Two chained regions on the same (i,j) — `extract_putlocalhom2`
        // emits this when the seed alignment has an internal indel.
        let text = "0 1 348 14356.319 0 337 0 337 k\n\
                    0 1 348 14356.319 338 347 338 347 k\n";
        let table = parse_hat3_seed(text, 2).expect("parse ok");
        assert_eq!(table.get(0, 1).len(), 2);
        assert_eq!(table.get(1, 0).len(), 2);
        let r = table.get(0, 1);
        assert_eq!(r[0].overlapaa, 348);
        assert_eq!(r[1].overlapaa, 348);
        assert_eq!(r[0].opt, 14356.319);
        assert_eq!(r[1].opt, 14356.319);
    }
}
