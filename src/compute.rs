//! Window evaluation: per activity, the best of each metric at each window length, optionally among
//! windows passing range filters. Lengths are seconds (time basis) or metres (distance basis).

use crate::prep::{Segment, Sport};
use rayon::prelude::*;

/// Metrics, in output order.
/// p power (W), v speed (m/s), g net gradient, h heart rate (bpm), a vertical speed (m/s), x grade-adjusted speed (m/s),
/// c cadence over the whole window (cycles/min, coasting and stops count as zero), k crank torque (N·m: work while
/// pedaling per radian of pedal revolution, at a pedaling cadence of at least 30/min), l distance per cycle (m: step length), pc cadence while pedaling or stepping
/// (cycles/min), ps share of the window spent pedaling or stepping (0-1)
pub const METRICS: [&str; 11] = ["p", "v", "g", "h", "a", "x", "c", "k", "l", "pc", "ps"];
pub const M: usize = METRICS.len();
const P: usize = 0;
const V: usize = 1;
const G: usize = 2;
const H: usize = 3;
const A: usize = 4;
const X: usize = 5;
const C: usize = 6;
const K: usize = 7;
const L: usize = 8;
const PC: usize = 9;
const PS: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Basis {
    Time,
    Dist,
}

/// What a filter tests: one of the window's own averages, or where it starts.
#[derive(Clone, Copy, Debug)]
pub enum Dim {
    Metric(usize),
    /// seconds into the activity at the window's start
    T0,
    /// metres into the activity at the window's start
    D0,
}

/// lo <= value <= hi; NaN bounds are open. A window whose value is unknown fails.
#[derive(Clone, Copy, Debug)]
pub struct Filter {
    pub dim: Dim,
    pub lo: f64,
    pub hi: f64,
}

impl Filter {
    #[inline]
    fn pass(&self, v: f64) -> bool {
        !v.is_nan() && (self.lo.is_nan() || v >= self.lo) && (self.hi.is_nan() || v <= self.hi)
    }
}

/// Bins of one dimension: a window falls into bin i when edges[i] <= value < edges[i + 1].
#[derive(Clone, Debug)]
pub struct Group {
    pub dim: Dim,
    pub edges: Vec<f64>,
}

impl Group {
    #[inline]
    fn bin(&self, v: f64) -> Option<usize> {
        if v.is_nan() {
            return None;
        }
        let i = self.edges.partition_point(|&e| e <= v);
        if i == 0 || i >= self.edges.len() {
            None
        } else {
            Some(i - 1)
        }
    }

    pub fn n(&self) -> usize {
        self.edges.len().saturating_sub(1)
    }
}

/// Per activity: best value (NaN = none) and its start (seconds into the activity), [group][len][metric].
pub struct ActResult {
    pub y: Vec<f32>,
    pub t0: Vec<i32>,
}

#[inline]
fn eval(s: &Segment, i: usize, j: usize, dt: f64, dd: f64, basis_len: f64, basis: Basis, sp: &Sport, out: &mut [f64; M]) {
    out[P] = if s.cp.is_empty() { f64::NAN } else { (s.cp[j] - s.cp[i]) / dt };
    out[V] = dd / dt;
    let hn = s.chn[j] - s.chn[i];
    out[H] = if hn >= 0.8 * dt { (s.ch[j] - s.ch[i]) / hn } else { f64::NAN };
    let (ai, aj) = (s.ca[i], s.ca[j]);
    let alt_ok = s.has_alt && !ai.is_nan() && !aj.is_nan();
    out[G] = if alt_ok && dd >= 20f64.max(0.3 * dt) {
        let g = (aj - ai) / dd;
        if g.abs() <= sp.grade_cap { g } else { f64::NAN }
    } else {
        f64::NAN
    };
    let long_enough_a = match basis {
        Basis::Time => basis_len >= sp.ascent_min,
        Basis::Dist => true,
    };
    out[A] = if alt_ok && long_enough_a { (s.cg[j] - s.cg[i]) / dt } else { f64::NAN };
    let long_enough_x = match basis {
        Basis::Time => basis_len >= sp.gap_min_time,
        Basis::Dist => basis_len >= sp.gap_min_dist,
    };
    out[X] = if !s.cx.is_empty() && long_enough_x {
        let x = (s.cx[j] - s.cx[i]) / dt;
        if x <= sp.speed_cap { x } else { f64::NAN }
    } else {
        f64::NAN
    };
    if s.crev.is_empty() {
        for m in [C, K, L, PC, PS] {
            out[m] = f64::NAN;
        }
    } else {
        let rev = s.crev[j] - s.crev[i];
        let ped = s.cped[j] - s.cped[i];
        out[C] = rev / dt * 60.0;
        out[PS] = ped / dt;
        // cadence while pedaling needs enough pedaling to mean something
        out[PC] = if ped >= 5f64.max(0.25 * dt) { rev / ped * 60.0 } else { f64::NAN };
        let enough = rev >= 1.0;
        // torque: work while pedaling per radian of revolution, only at a real pedaling cadence (>= 30/min)
        out[K] = if enough && !s.cpw.is_empty() && out[PC] >= 30.0 {
            (s.cpw[j] - s.cpw[i]) / (2.0 * std::f64::consts::PI * rev)
        } else {
            f64::NAN
        };
        out[L] = if enough { dd / rev } else { f64::NAN };
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn consider(
    s: &Segment,
    i: usize,
    j: usize,
    dt: f64,
    dd: f64,
    len: f64,
    basis: Basis,
    sp: &Sport,
    filters: &[Filter],
    group: Option<&Group>,
    k_n: usize,
    k: usize,
    res: &mut ActResult,
) {
    let mut w = [0.0; M];
    eval(s, i, j, dt, dd, len, basis, sp, &mut w);
    let value = |dim: Dim| match dim {
        Dim::Metric(m) => w[m],
        Dim::T0 => s.t_off + i as f64,
        Dim::D0 => s.d_off + s.cd[i],
    };
    for f in filters {
        if !f.pass(value(f.dim)) {
            return;
        }
    }
    let g = match group {
        Some(gr) => match gr.bin(value(gr.dim)) {
            Some(b) => b,
            None => return,
        },
        None => 0,
    };
    let base = (g * k_n + k) * M;
    for m in 0..M {
        let v = w[m] as f32;
        let cur = res.y[base + m];
        if v > cur || (cur.is_nan() && !v.is_nan()) {
            res.y[base + m] = v;
            res.t0[base + m] = (s.t_off + i as f64) as i32;
        }
    }
}

/// Evaluate every window of every length in one activity's segments.
///
/// `eps` sets the start stride: windows of length L start every max(1, floor(eps * L)) seconds
/// (for distance windows, every max(1, floor(eps * L / speed_cap)) seconds). eps = 0 is exhaustive.
#[allow(clippy::too_many_arguments)]
pub fn activity(
    segs: &[&Segment],
    basis: Basis,
    lengths: &[f64],
    eps: f64,
    filters: &[Filter],
    group: Option<&Group>,
    sp: &Sport,
) -> ActResult {
    let k_n = lengths.len();
    let g_n = group.map_or(1, |g| g.n());
    let mut res = ActResult { y: vec![f32::NAN; g_n * k_n * M], t0: vec![-1; g_n * k_n * M] };
    for s in segs {
        let n = s.n();
        match basis {
            Basis::Time => {
                for (k, &len) in lengths.iter().enumerate() {
                    let l = len as usize;
                    if l == 0 || l >= n {
                        continue;
                    }
                    let stride = ((eps * len) as usize).max(1);
                    let dt = len;
                    let last = n - 1 - l;
                    let mut i = 0;
                    while i <= last {
                        let j = i + l;
                        let dd = s.cd[j] - s.cd[i];
                        consider(s, i, j, dt, dd, len, basis, sp, filters, group, k_n, k, &mut res);
                        i += stride;
                    }
                    // the window ending at the segment's end: with it, every window of a shorter length lies
                    // inside some evaluated window of a slightly longer one, which is what bounds the stride error
                    if stride > 1 && last % stride != 0 {
                        consider(s, last, n - 1, dt, s.cd[n - 1] - s.cd[last], len, basis, sp, filters, group, k_n, k, &mut res);
                    }
                }
            }
            Basis::Dist => {
                let total = s.cd[n - 1];
                for (k, &len) in lengths.iter().enumerate() {
                    if len > total {
                        continue;
                    }
                    // starts spaced by eps times the current window's duration, so a skipped window lies inside an
                    // evaluated one that is at most eps longer (striding by distance would skip the end of a stop)
                    let stride = if eps > 0.0 { 2 } else { 1 }; // only decides whether the end window is needed
                    let mut j = 0;
                    let mut i = 0;
                    let mut last_i = usize::MAX;
                    while i < n {
                        if j < i {
                            j = i;
                        }
                        let target = s.cd[i] + len;
                        while j < n && s.cd[j] < target {
                            j += 1;
                        }
                        if j == n {
                            break;
                        }
                        let dt = (j - i) as f64;
                        consider(s, i, j, dt, len, len, basis, sp, filters, group, k_n, k, &mut res);
                        last_i = i;
                        i += ((eps * dt) as usize).max(1);
                    }
                    // the latest start whose window still fits, as for time windows
                    if stride > 1 && last_i != usize::MAX {
                        let end = s.cd[n - 1] - len;
                        let i = s.cd.partition_point(|&c| c <= end).saturating_sub(1);
                        if i > last_i {
                            let j = s.cd.partition_point(|&c| c < s.cd[i] + len);
                            if j < n {
                                consider(s, i, j, (j - i) as f64, len, len, basis, sp, filters, group, k_n, k, &mut res);
                            }
                        }
                    }
                }
            }
        }
    }
    res
}

/// The selected activities in parallel, results in the order of `acts`. `by_act[a]` lists activity a's segments.
#[allow(clippy::too_many_arguments)]
pub fn all(
    by_act: &[Vec<&Segment>],
    acts: &[usize],
    basis: Basis,
    lengths: &[f64],
    eps: f64,
    filters: &[Filter],
    group: Option<&Group>,
    sp: &Sport,
) -> Vec<ActResult> {
    // longest activities first so the stragglers start early
    let mut order: Vec<usize> = (0..acts.len()).collect();
    order.sort_by_key(|&o| std::cmp::Reverse(by_act[acts[o]].iter().map(|s| s.n()).sum::<usize>()));
    let mut done: Vec<(usize, ActResult)> = order
        .par_iter()
        .map(|&o| (o, activity(&by_act[acts[o]], basis, lengths, eps, filters, group, sp)))
        .collect();
    done.sort_by_key(|(o, _)| *o);
    done.into_iter().map(|(_, r)| r).collect()
}
