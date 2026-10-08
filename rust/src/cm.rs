//! Covariance-model CYK alignment (Rust port of the pure-Python reference).
//!
//! Mirrors `organelleverse.annotation.cmsearch.cyk.cyk_align` exactly so the two
//! paths return identical coordinates and scores. The model is passed as flat
//! arrays (built once in Python) to avoid duplicating the .cm parser here.
//!
//! State type codes: E=0 S=1 D=2 ML=3 MR=4 IL=5 IR=6 MP=7 B=8 EL=9.

const E: u8 = 0;
const S: u8 = 1;
const D: u8 = 2;
const ML: u8 = 3;
const MR: u8 = 4;
const IL: u8 = 5;
const IR: u8 = 6;
const MP: u8 = 7;
const B: u8 = 8;
const EL: u8 = 9;

fn encode(seq: &[u8]) -> Vec<i32> {
    seq.iter()
        .map(|&c| match c {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ => -1,
        })
        .collect()
}

pub struct Cm<'a> {
    pub types: &'a [u8],
    pub cfirst: &'a [i32],
    pub cnum: &'a [i32],
    pub trans_flat: &'a [f64],
    pub trans_off: &'a [usize],
    pub emit_flat: &'a [f64],
    pub emit_off: &'a [usize],
    // Optional per-state node index and per-node consensus columns (for traceback).
    pub state_node: &'a [i32],
    pub node_lcol: &'a [i32],
    pub node_rcol: &'a [i32],
}

/// Fill the CYK score table; returns (dp, w) where dp[v][i*w + j] is the score.
fn forward_dp(cm: &Cm, x: &[i32]) -> (Vec<Vec<f64>>, usize) {
    let l = x.len();
    let n = cm.types.len();
    let neg = f64::NEG_INFINITY;
    let w = l + 2;
    let mut dp: Vec<Vec<f64>> = vec![vec![neg; w * w]; n];

    for v in (0..n).rev() {
        let t = cm.types[v];
        let trans = &cm.trans_flat[cm.trans_off[v]..cm.trans_off[v + 1]];
        let emit = &cm.emit_flat[cm.emit_off[v]..cm.emit_off[v + 1]];
        let cfirst = cm.cfirst[v];
        let cnum = cm.cnum[v];

        // Precompute children (indices) for non-bifurcation states.
        let mut i = l + 1;
        while i >= 1 {
            let mut j = i - 1;
            while j <= l {
                let d = j as i64 - i as i64 + 1;
                let val: f64 = match t {
                    E | EL => {
                        if d == 0 {
                            0.0
                        } else {
                            neg
                        }
                    }
                    S | D => {
                        let mut best = neg;
                        for k in 0..cnum as usize {
                            let c = (cfirst + k as i32) as usize;
                            let cand = trans[k] + dp[c][i * w + j];
                            if cand > best {
                                best = cand;
                            }
                        }
                        best
                    }
                    ML | IL => {
                        if d < 1 {
                            neg
                        } else {
                            let a = x_at(&x, i);
                            let es = if a >= 0 { emit[a as usize] } else { neg };
                            let mut best = neg;
                            for k in 0..cnum as usize {
                                let c = (cfirst + k as i32) as usize;
                                let cand = trans[k] + dp[c][(i + 1) * w + j];
                                if cand > best {
                                    best = cand;
                                }
                            }
                            es + best
                        }
                    }
                    MR | IR => {
                        if d < 1 {
                            neg
                        } else {
                            let b = x_at(&x, j);
                            let es = if b >= 0 { emit[b as usize] } else { neg };
                            let mut best = neg;
                            for k in 0..cnum as usize {
                                let c = (cfirst + k as i32) as usize;
                                let cand = trans[k] + dp[c][i * w + (j - 1)];
                                if cand > best {
                                    best = cand;
                                }
                            }
                            es + best
                        }
                    }
                    MP => {
                        if d < 2 {
                            neg
                        } else {
                            let a = x_at(&x, i);
                            let b = x_at(&x, j);
                            let es = if a >= 0 && b >= 0 {
                                emit[(4 * a + b) as usize]
                            } else {
                                neg
                            };
                            let mut best = neg;
                            for k in 0..cnum as usize {
                                let c = (cfirst + k as i32) as usize;
                                let cand = trans[k] + dp[c][(i + 1) * w + (j - 1)];
                                if cand > best {
                                    best = cand;
                                }
                            }
                            es + best
                        }
                    }
                    B => {
                        let left = cfirst as usize;
                        let right = cnum as usize;
                        let mut best = neg;
                        let mut k = i - 1;
                        while k <= j {
                            let cand = dp[left][i * w + k] + dp[right][(k + 1) * w + j];
                            if cand > best {
                                best = cand;
                            }
                            k += 1;
                        }
                        best
                    }
                    _ => neg,
                };
                dp[v][i * w + j] = val;
                j += 1;
            }
            i -= 1;
        }
    }
    (dp, w)
}

fn best_root(dp: &[Vec<f64>], w: usize, l: usize) -> (usize, usize, f64) {
    let root = &dp[0];
    let mut best_score = f64::NEG_INFINITY;
    let mut best_i = 0usize;
    let mut best_j = 0usize;
    for i in 1..=l {
        for j in i..=l {
            let s = root[i * w + j];
            if s > best_score {
                best_score = s;
                best_i = i;
                best_j = j;
            }
        }
    }
    (best_i, best_j, best_score)
}

/// Best CM alignment to a subsequence of `seq`; returns (start, end, score)
/// with 1-based inclusive coordinates, or None if no alignment beats `min_score`.
pub fn cyk_best(cm: &Cm, seq: &[u8], min_score: f64) -> Option<(usize, usize, f64)> {
    let x = encode(seq);
    let l = x.len();
    let (dp, w) = forward_dp(cm, &x);
    let (bi, bj, bs) = best_root(&dp, w, l);
    if bi == 0 || bs <= min_score {
        None
    } else {
        Some((bi, bj, bs))
    }
}

/// Best CM alignment with traceback: returns (start, end, score, alignment)
/// where alignment is (consensus_column, seq_pos) for match emissions, sorted by
/// consensus column. Mirrors the pure-Python `cyk_trace`.
pub fn cyk_trace_best(cm: &Cm, seq: &[u8]) -> Option<(usize, usize, f64, Vec<(i32, i32)>)> {
    let x = encode(seq);
    let l = x.len();
    let neg = f64::NEG_INFINITY;
    let (dp, w) = forward_dp(cm, &x);
    let (bi, bj, bs) = best_root(&dp, w, l);
    if bi == 0 {
        return None;
    }

    let eps = 1e-6;
    let mut emissions: Vec<(i32, i32)> = Vec::new();
    let mut stack: Vec<(usize, usize, usize)> = vec![(0, bi, bj)];
    while let Some((v, i, j)) = stack.pop() {
        let t = cm.types[v];
        let trans = &cm.trans_flat[cm.trans_off[v]..cm.trans_off[v + 1]];
        let emit = &cm.emit_flat[cm.emit_off[v]..cm.emit_off[v + 1]];
        let cfirst = cm.cfirst[v];
        let cnum = cm.cnum[v];
        let node = cm.state_node[v];
        let (lcol, rcol) = if node >= 0 {
            (cm.node_lcol[node as usize], cm.node_rcol[node as usize])
        } else {
            (-1, -1)
        };
        let target = dp[v][i * w + j];

        match t {
            E | EL => {}
            S | D => {
                for k in 0..cnum as usize {
                    let c = (cfirst + k as i32) as usize;
                    if (trans[k] + dp[c][i * w + j] - target).abs() < eps {
                        stack.push((c, i, j));
                        break;
                    }
                }
            }
            ML | IL => {
                let a = x_at(&x, i);
                let es = if a >= 0 { emit[a as usize] } else { neg };
                if t == ML && lcol >= 0 {
                    emissions.push((lcol, i as i32));
                }
                for k in 0..cnum as usize {
                    let c = (cfirst + k as i32) as usize;
                    if (es + trans[k] + dp[c][(i + 1) * w + j] - target).abs() < eps {
                        stack.push((c, i + 1, j));
                        break;
                    }
                }
            }
            MR | IR => {
                let b = x_at(&x, j);
                let es = if b >= 0 { emit[b as usize] } else { neg };
                if t == MR && rcol >= 0 {
                    emissions.push((rcol, j as i32));
                }
                for k in 0..cnum as usize {
                    let c = (cfirst + k as i32) as usize;
                    if (es + trans[k] + dp[c][i * w + (j - 1)] - target).abs() < eps {
                        stack.push((c, i, j - 1));
                        break;
                    }
                }
            }
            MP => {
                let a = x_at(&x, i);
                let b = x_at(&x, j);
                let es = if a >= 0 && b >= 0 {
                    emit[(4 * a + b) as usize]
                } else {
                    neg
                };
                if lcol >= 0 {
                    emissions.push((lcol, i as i32));
                }
                if rcol >= 0 {
                    emissions.push((rcol, j as i32));
                }
                for k in 0..cnum as usize {
                    let c = (cfirst + k as i32) as usize;
                    if (es + trans[k] + dp[c][(i + 1) * w + (j - 1)] - target).abs() < eps {
                        stack.push((c, i + 1, j - 1));
                        break;
                    }
                }
            }
            B => {
                let left = cfirst as usize;
                let right = cnum as usize;
                let mut k = i - 1;
                while k <= j {
                    if (dp[left][i * w + k] + dp[right][(k + 1) * w + j] - target).abs() < eps {
                        stack.push((left, i, k));
                        stack.push((right, k + 1, j));
                        break;
                    }
                    k += 1;
                }
            }
            _ => {}
        }
    }
    emissions.sort_by_key(|e| e.0);
    Some((bi, bj, bs, emissions))
}

#[inline]
fn x_at(x: &[i32], pos1: usize) -> i32 {
    // pos1 is 1-based; guard against the empty-span sentinels.
    if pos1 == 0 || pos1 > x.len() {
        -1
    } else {
        x[pos1 - 1]
    }
}
