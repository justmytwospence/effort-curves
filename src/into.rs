//! The frontier over a property of windows: for windows of one length, the best of a metric among windows at least as
//! hard as x on a dimension with a harder direction, as a function of x. The dimension is where a window starts (seconds
//! or metres in, `t0`/`d0`: later is harder) or one of the window's own metrics (heart rate: lower is harder for the
//! same pace or power; gradient: steeper). It is a staircase read from the hard end: a window is on it when it beats
//! every window that is harder still. It is exact: every start of every activity is a candidate.
//!
//! Direction `Ge` means larger x is harder; `Le` means smaller x is harder (the dimension is mirrored inside, so the
//! same staircase applies, and mirrored back on the way out).
//!
//! Consecutive staircase points from one activity whose windows overlap in time are one **effort**: it holds the
//! frontier from the previous effort's last start (exclusive) up to its own last start.

use crate::compute::{eval_one, passes, Basis, Dim, Filter};
use crate::prep::{Segment, Sport};
use rayon::prelude::*;

/// A window: where it starts on the x dimension, its value, its start (s into the activity) and end (s).
#[derive(Clone, Copy, Debug)]
struct Win {
    x: f64,
    y: f32,
    t0: f64,
    t1: f64,
}

/// One activity's windows of `len` that pass the filters and are on its own staircase (beaten by no later start),
/// in ascending x. A window dominated within its activity is dominated in any set containing it.
#[allow(clippy::too_many_arguments)]
fn own_staircase(segs: &[&Segment], basis: Basis, len: f64, m: usize, dim: Dim, sign: f64, filters: &[Filter], sp: &Sport) -> Vec<Win> {
    let mut all: Vec<Win> = Vec::new();
    for s in segs {
        let n = s.n();
        // the window's x, mirrored for a smaller-is-harder dimension
        let x_of = |i: usize, j: usize, dt: f64, dd: f64| {
            sign * match dim {
                Dim::T0 => s.t_off + i as f64,
                Dim::D0 => s.d_off + s.cd[i],
                Dim::Metric(d) => eval_one(s, i, j, dt, dd, len, basis, sp, d),
            }
        };
        match basis {
            Basis::Time => {
                let l = len as usize;
                if l == 0 || l >= n {
                    continue;
                }
                for i in 0..n - l {
                    let j = i + l;
                    let dd = s.cd[j] - s.cd[i];
                    if passes(s, i, j, len, dd, len, basis, sp, filters) {
                        let y = eval_one(s, i, j, len, dd, len, basis, sp, m) as f32;
                        let x = x_of(i, j, len, dd);
                        if !y.is_nan() && !x.is_nan() {
                            all.push(Win { x, y, t0: s.t_off + i as f64, t1: s.t_off + j as f64 });
                        }
                    }
                }
            }
            Basis::Dist => {
                if len > s.cd[n - 1] {
                    continue;
                }
                let mut j = 0;
                for i in 0..n {
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
                    if passes(s, i, j, dt, len, len, basis, sp, filters) {
                        let y = eval_one(s, i, j, dt, len, len, basis, sp, m) as f32;
                        let x = x_of(i, j, dt, len);
                        if !y.is_nan() && !x.is_nan() {
                            all.push(Win { x, y, t0: s.t_off + i as f64, t1: s.t_off + j as f64 });
                        }
                    }
                }
            }
        }
    }
    staircase(&mut all)
}

/// The staircase of a set of windows: those beating every window with a larger x (ties in x: the best of them), in
/// ascending x.
fn staircase(w: &mut [Win]) -> Vec<Win> {
    w.sort_by(|a, b| b.x.partial_cmp(&a.x).unwrap().then(b.y.partial_cmp(&a.y).unwrap()));
    let mut out = Vec::new();
    let mut held = f32::NAN;
    for v in w.iter() {
        if held.is_nan() || v.y > held {
            held = v.y;
            out.push(*v);
        }
    }
    out.reverse();
    out
}

/// One effort of a frontier over starts.
#[derive(Clone, Debug)]
pub struct IntoEffort {
    pub set: usize,
    pub effort: usize,
    pub row: usize,
    /// its hardest staircase window: start (s into the activity), x, value
    pub t0: i32,
    pub x: f64,
    pub y: f32,
    /// it holds the frontier from this x (the previous effort's hardest x, exclusive; NaN for the first) to its own, at
    /// from_y next to it
    pub from_x: f64,
    pub from_y: f32,
}

/// The frontier of each set (lists of rows; `rows[r]` is the activity of row r), as efforts and staircase points
/// ((set, effort, x, y), ascending x within a set).
#[allow(clippy::too_many_arguments)]
pub fn all(
    by_act: &[Vec<&Segment>],
    rows: &[usize],
    sets: &[Vec<usize>],
    basis: Basis,
    len: f64,
    m: usize,
    dim: Dim,
    larger_is_harder: bool,
    filters: &[Filter],
    sp: &Sport,
) -> (Vec<IntoEffort>, Vec<(usize, usize, f64, f32)>) {
    let sign = if larger_is_harder { 1.0 } else { -1.0 };
    // each activity's own staircase once, shared by every set it is in
    let own: Vec<Vec<Win>> = rows.par_iter().map(|&a| own_staircase(&by_act[a], basis, len, m, dim, sign, filters, sp)).collect();
    let per_set: Vec<(Vec<IntoEffort>, Vec<(usize, usize, f64, f32)>)> = sets
        .par_iter()
        .enumerate()
        .map(|(si, set)| {
            let mut w: Vec<(Win, usize)> = set.iter().flat_map(|&r| own[r].iter().map(move |v| (*v, r))).collect();
            w.sort_by(|a, b| b.0.x.partial_cmp(&a.0.x).unwrap().then(b.0.y.partial_cmp(&a.0.y).unwrap()));
            let mut stair: Vec<(Win, usize)> = Vec::new();
            let mut held = f32::NAN;
            for (v, r) in w {
                if held.is_nan() || v.y > held {
                    held = v.y;
                    stair.push((v, r));
                }
            }
            stair.reverse();
            // efforts: consecutive points of one activity whose windows overlap in time
            let mut eff: Vec<IntoEffort> = Vec::new();
            let mut pts = Vec::with_capacity(stair.len());
            let mut span = (0.0, 0.0);
            for (v, r) in stair {
                let same = eff.last().is_some_and(|e| e.row == r) && v.t0 < span.1 && v.t1 > span.0;
                if same {
                    let e = eff.last_mut().unwrap();
                    e.t0 = v.t0 as i32;
                    e.x = v.x;
                    e.y = v.y;
                    span = (span.0.min(v.t0), span.1.max(v.t1));
                } else {
                    let (from_x, from_y) = match eff.last() {
                        Some(p) => (p.x, v.y),
                        None => (f64::NAN, f32::NAN),
                    };
                    eff.push(IntoEffort { set: si, effort: eff.len(), row: r, t0: v.t0 as i32, x: v.x, y: v.y, from_x, from_y });
                    span = (v.t0, v.t1);
                }
                pts.push((si, eff.len() - 1, v.x, v.y));
            }
            (eff, pts)
        })
        .collect();
    // back to the dimension's own sense (a mirrored dimension's points come out in descending x)
    let mut e_out = Vec::new();
    let mut p_out = Vec::new();
    for (e, p) in per_set {
        e_out.extend(e.into_iter().map(|mut v| {
            v.x *= sign;
            v.from_x *= sign;
            v
        }));
        p_out.extend(p.into_iter().map(|(a, b, x, y)| (a, b, x * sign, y)));
    }
    (e_out, p_out)
}
