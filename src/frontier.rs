//! The frontier as efforts: which stretch of which activity holds the best of one metric over which range of lengths.
//!
//! Input is a grid of per-activity bests (`compute`'s output for one metric, not enveloped). For a set of activities
//! the frontier at length L is the best window of length L or longer (held for at least L). A length is *efficient*
//! when its best window is the frontier itself, not lifted by a longer one. Consecutive efficient lengths whose
//! windows come from the same activity and overlap in time are one **effort**: its full length is its longest window,
//! and it holds the frontier from the previous effort's full length up to its own.
//!
//! With `refine`, the frontier is made exact to the second (time) or metre (distance). Every grid step that could
//! hide anything is evaluated exhaustively at every integer length, for the activities that could matter there, and
//! the efforts are rebuilt from that exact sequence. A step is evaluated when it holds a handoff (or follows one), and,
//! where a strict bound exists, when an activity other than the step's owner could beat the frontier inside it:
//! - time basis, metrics with a non-negative running total (power, speed, grade-adjusted speed, vertical speed,
//!   cadence): a window of length L lies inside one of length L' > L, so best(L) <= best(L') * L' / L (and an
//!   activity that ends inside the step is bounded by its largest total over L);
//! - distance basis, speed: a window of L metres contains one of L' < L metres, which takes at least L' / best(L')
//!   seconds, so best(L) <= best(L') * L / L'.
//! Without a bound (heart rate, gradient, torque, step length, pedaling cadence and share, and every metric but speed
//! on the distance basis) only the handoffs are evaluated: they are exact, but an effort of a third activity that
//! holds less than one grid step and lies between two grid lengths is not looked for.

use crate::compute::{activity, activity_metric, max_total, Basis, Filter, Group, M};
use crate::prep::{Segment, Sport};
use rayon::prelude::*;

/// Metrics with a non-negative running total (p v a x c), where the containment bound holds on the time basis.
const BOUNDED: [usize; 5] = [0, 1, 4, 5, 6];

/// One effort of a frontier.
#[derive(Clone, Debug)]
pub struct Effort {
    pub set: usize,
    pub cell: usize,
    /// order along the frontier, shortest first
    pub effort: usize,
    /// row of the input arrays (the activity)
    pub row: usize,
    /// start of its full window, seconds into the activity
    pub t0: i32,
    /// its full length (s or m)
    pub len: f64,
    /// its value at the full length
    pub y: f32,
    /// it holds the frontier for lengths above this one (the previous effort's full length; NaN for the first)
    pub from_len: f64,
    /// the frontier just above from_len (NaN for the first)
    pub from_y: f32,
}

/// Per-activity grid of one metric: y and t0 indexed [row][cell][k].
pub struct Grid<'a> {
    pub y: &'a [f32],
    pub t0: &'a [i32],
    pub cells: usize,
    pub k: usize,
}

impl Grid<'_> {
    #[inline]
    fn at(&self, row: usize, cell: usize, k: usize) -> (f32, i32) {
        let i = (row * self.cells + cell) * self.k + k;
        (self.y[i], self.t0[i])
    }
}

/// The time a window covers, in seconds into the activity: [start, end).
fn span(segs: &[&Segment], basis: Basis, t0: i32, len: f64) -> (f64, f64) {
    let t = t0 as f64;
    match basis {
        Basis::Time => (t, t + len),
        Basis::Dist => {
            for s in segs {
                let i = t - s.t_off;
                if i >= 0.0 && (i as usize) < s.n() {
                    let i = i as usize;
                    let j = s.cd.partition_point(|&c| c < s.cd[i] + len);
                    return (t, s.t_off + j.min(s.n() - 1) as f64);
                }
            }
            (t, t + 1.0)
        }
    }
}

#[inline]
fn better(v: f32, cur: f32) -> bool {
    !v.is_nan() && (cur.is_nan() || v > cur)
}

/// A length's best window over the set: (length, value, row, t0).
type Point = (f64, f32, usize, i32);

/// Efforts from a sequence of points (ascending length): envelope from the long end, efficient points, grouped.
fn group(by_act: &[Vec<&Segment>], rows: &[usize], basis: Basis, pts: &[Point]) -> Vec<Effort> {
    let mut eff: Vec<bool> = vec![false; pts.len()];
    let mut held = f32::NAN;
    for i in (0..pts.len()).rev() {
        if better(pts[i].1, held) {
            held = pts[i].1;
            eff[i] = true;
        }
    }
    let mut out: Vec<Effort> = Vec::new();
    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut prev_held_above: Vec<f32> = vec![f32::NAN; pts.len()]; // the frontier just above each point
    let mut h = f32::NAN;
    for i in (0..pts.len()).rev() {
        prev_held_above[i] = h;
        if better(pts[i].1, h) {
            h = pts[i].1;
        }
    }
    let mut last_i: Option<usize> = None;
    for (i, &(len, v, r, t0)) in pts.iter().enumerate() {
        if !eff[i] {
            continue;
        }
        let s = span(&by_act[rows[r]], basis, t0, len);
        if let (Some(d), Some(sp)) = (out.last_mut(), spans.last_mut()) {
            if d.row == r && s.0 < sp.1 && s.1 > sp.0 {
                d.len = len;
                d.t0 = t0;
                d.y = v;
                *sp = (sp.0.min(s.0), sp.1.max(s.1));
                last_i = Some(i);
                continue;
            }
        }
        let (from_len, from_y) = match last_i {
            Some(j) => (pts[j].0, prev_held_above[j]),
            None => (f64::NAN, f32::NAN),
        };
        out.push(Effort { set: 0, cell: 0, effort: out.len(), row: r, t0, len, y: v, from_len, from_y });
        spans.push(s);
        last_i = Some(i);
    }
    out
}

/// What refining a step needs.
struct Ctx<'a> {
    by_act: &'a [Vec<&'a Segment>],
    rows: &'a [usize],
    basis: Basis,
    metric: usize,
    filters: &'a [Filter],
    groups: &'a [Group],
    sp: &'a Sport,
    cell: usize,
    time_bound: bool,
    dist_bound: bool,
}

/// Steps with at most this many lengths inside are evaluated at every one.
const DIRECT: usize = 24;

impl Ctx<'_> {
    /// One row's best at each length (exhaustive): (value, t0).
    fn eval(&self, r: usize, ls: &[f64]) -> Vec<(f32, i32)> {
        let segs = &self.by_act[self.rows[r]];
        if self.filters.is_empty() && self.groups.is_empty() {
            let (y, t0) = activity_metric(segs, self.basis, ls, self.metric, self.sp);
            y.into_iter().zip(t0).collect()
        } else {
            let res = activity(segs, self.basis, ls, 0.0, self.filters, self.groups, self.sp);
            (0..ls.len()).map(|g| { let i = (self.cell * ls.len() + g) * M + self.metric; (res.y[i], res.t0[i]) }).collect()
        }
    }

    /// Could row r beat `floor` somewhere strictly between lengths a and b, given its values ya, yb there?
    fn could(&self, r: usize, a: f64, b: f64, ya: f32, yb: f32, floor: f32) -> bool {
        if !(self.time_bound || self.dist_bound) || ya.is_nan() {
            return false;
        }
        if floor.is_nan() {
            return true;
        }
        if self.dist_bound {
            return (ya as f64 * b / a) as f32 > floor;
        }
        if yb.is_nan() {
            // it ends inside the step: bounded by its largest total over the step's shortest length
            return max_total(&self.by_act[self.rows[r]], self.metric).is_none_or(|tot| (tot / a) as f32 > floor);
        }
        (yb as f64 * b / a) as f32 > floor
    }

    /// Same stretch: windows of the same row that overlap in time.
    fn same(&self, x: (usize, i32, f64), y: (usize, i32, f64)) -> bool {
        if x.0 != y.0 {
            return false;
        }
        let segs = &self.by_act[self.rows[x.0]];
        let (a, b) = (span(segs, self.basis, x.1, x.2), span(segs, self.basis, y.1, y.2));
        a.0 < b.1 && a.1 > b.0
    }

    /// The exact best of `cands` at the lengths strictly between lo and hi that matter, given their values at both
    /// ends ((row, value, t0)) and the frontier at hi.
    fn refine(&self, lo: f64, hi: f64, cands: &[usize], at_lo: &[(usize, f32, i32)], at_hi: &[(usize, f32, i32)], env_hi: f32) -> Vec<Point> {
        let inside: Vec<f64> = ((lo.floor() as i64 + 1)..(hi.ceil() as i64)).map(|v| v as f64).filter(|&v| v > lo && v < hi).collect();
        if inside.is_empty() || cands.is_empty() {
            return Vec::new();
        }
        let best_of = |ls: &[f64], vals: &[Vec<(f32, i32)>]| -> Vec<Point> {
            (0..ls.len())
                .map(|g| {
                    let mut p: Point = (ls[g], f32::NAN, usize::MAX, -1);
                    for (ci, &r) in cands.iter().enumerate() {
                        if better(vals[ci][g].0, p.1) {
                            p = (ls[g], vals[ci][g].0, r, vals[ci][g].1);
                        }
                    }
                    p
                })
                .collect()
        };
        if inside.len() <= DIRECT {
            let vals: Vec<Vec<(f32, i32)>> = cands.iter().map(|&r| self.eval(r, &inside)).collect();
            return best_of(&inside, &vals);
        }
        // a few lengths across the step, then each piece between them on the same terms
        let s = (inside.len() as f64).sqrt().ceil() as usize;
        let subs: Vec<f64> = inside.iter().copied().skip(s - 1).step_by(s).collect();
        let vals: Vec<Vec<(f32, i32)>> = cands.iter().map(|&r| self.eval(r, &subs)).collect();
        let mut out = best_of(&subs, &vals);
        // values at every boundary (lo, subs..., hi), per candidate
        let nb = subs.len() + 2;
        let bl: Vec<f64> = std::iter::once(lo).chain(subs.iter().copied()).chain(std::iter::once(hi)).collect();
        let val = |ci: usize, bi: usize| -> (f32, i32) {
            if bi == 0 {
                (at_lo[ci].1, at_lo[ci].2)
            } else if bi == nb - 1 {
                (at_hi[ci].1, at_hi[ci].2)
            } else {
                vals[ci][bi - 1]
            }
        };
        // the frontier at each boundary from the long end, and who holds it
        let mut env = vec![(f32::NAN, usize::MAX, -1i32, 0.0f64); nb];
        // at hi: the best candidate there, unless the frontier beyond it is higher (held by a longer effort)
        let mut h = (f32::NAN, usize::MAX, -1i32, hi);
        for ci in 0..cands.len() {
            let (v, t) = val(ci, nb - 1);
            if better(v, h.0) {
                h = (v, cands[ci], t, hi);
            }
        }
        if better(env_hi, h.0) {
            h = (env_hi, usize::MAX, -1, hi);
        }
        env[nb - 1] = h;
        for bi in (0..nb - 1).rev() {
            for ci in 0..cands.len() {
                let (v, t) = val(ci, bi);
                if better(v, h.0) {
                    h = (v, cands[ci], t, bl[bi]);
                }
            }
            env[bi] = h;
        }
        for bi in 0..nb - 1 {
            let (a, b) = (bl[bi], bl[bi + 1]);
            let floor = env[bi + 1].0;
            let mut next: Vec<usize> = Vec::new();
            for (ci, &r) in cands.iter().enumerate() {
                if self.could(r, a, b, val(ci, bi).0, val(ci, bi + 1).0, floor) {
                    next.push(r);
                }
            }
            let (oa, ob) = (env[bi], env[bi + 1]);
            let handoff = match (oa.1 == usize::MAX, ob.1 == usize::MAX) {
                (true, true) => false,
                (false, false) => !self.same((oa.1, oa.2, oa.3), (ob.1, ob.2, ob.3)),
                _ => true,
            };
            let others = next.iter().any(|&r| r != oa.1 && r != ob.1);
            if !(handoff || others) {
                continue;
            }
            for o in [oa.1, ob.1] {
                if o != usize::MAX && !next.contains(&o) && cands.contains(&o) {
                    next.push(o);
                }
            }
            let pick = |bi: usize| -> Vec<(usize, f32, i32)> {
                next.iter().map(|&r| { let ci = cands.iter().position(|&c| c == r).unwrap(); let (v, t) = val(ci, bi); (r, v, t) }).collect()
            };
            out.extend(self.refine(a, b, &next, &pick(bi), &pick(bi + 1), floor));
        }
        out
    }
}

/// The frontier of one set of rows in one cell.
#[allow(clippy::too_many_arguments)]
fn one(
    by_act: &[Vec<&Segment>],
    rows: &[usize],
    grid: &Grid,
    set: &[usize],
    cell: usize,
    lengths: &[f64],
    metric: usize,
    basis: Basis,
    filters: &[Filter],
    groups: &[Group],
    sp: &Sport,
    refine: bool,
) -> Vec<Effort> {
    let kn = lengths.len();
    // best over the set at each grid length
    let mut pts: Vec<Point> = Vec::with_capacity(kn);
    for (k, &len) in lengths.iter().enumerate() {
        let mut b: Point = (len, f32::NAN, usize::MAX, -1);
        for &r in set {
            let (v, t0) = grid.at(r, cell, k);
            if better(v, b.1) {
                b = (len, v, r, t0);
            }
        }
        pts.push(b);
    }
    let coarse = group(by_act, rows, basis, &pts);
    if !refine || kn < 2 {
        return coarse;
    }
    // the frontier at each grid length (held from the long end), the floor of what a step can hold
    let mut env = vec![f32::NAN; kn];
    let mut h = f32::NAN;
    for k in (0..kn).rev() {
        if better(pts[k].1, h) {
            h = pts[k].1;
        }
        env[k] = h;
    }
    // steps (k, k + 1) to evaluate: around every grid handoff, and wherever someone else could beat the frontier
    let mut step = vec![false; kn - 1];
    for e in &coarse[..coarse.len().saturating_sub(1)] {
        let k = lengths.partition_point(|&l| l < e.len);
        for s in [k, k + 1] {
            if s < kn - 1 {
                step[s] = true;
            }
        }
    }
    let time_bound = basis == Basis::Time && BOUNDED.contains(&metric);
    let dist_bound = basis == Basis::Dist && metric == 1;
    // rows to evaluate in each step (the owners of both ends, and anyone who could beat the frontier there)
    let mut who: Vec<Vec<usize>> = vec![Vec::new(); kn - 1];
    for k in 0..kn - 1 {
        let floor = env[k + 1];
        let owners = [pts[k].2, pts[k + 1].2];
        for &r in set {
            let (a, _) = grid.at(r, cell, k);
            let (b, _) = grid.at(r, cell, k + 1);
            let could = if !(time_bound || dist_bound) || a.is_nan() {
                false
            } else if floor.is_nan() {
                true
            } else if dist_bound {
                (a as f64 * lengths[k + 1] / lengths[k]) as f32 > floor
            } else if b.is_nan() {
                // it ends inside the step: bounded by its largest total over the step's shortest length
                max_total(&by_act[rows[r]], metric).is_none_or(|tot| (tot / lengths[k]) as f32 > floor)
            } else {
                (b as f64 * lengths[k + 1] / lengths[k]) as f32 > floor
            };
            if could && !owners.contains(&r) {
                step[k] = true;
                who[k].push(r);
            }
        }
        if step[k] {
            for o in owners {
                if o != usize::MAX && !who[k].contains(&o) {
                    who[k].push(o);
                }
            }
        }
    }
    // each evaluated step, refined in parallel: its candidates at every integer length (or, in a long step, at a few
    // and then only where something could still change)
    let cx = Ctx { by_act, rows, basis, metric, filters, groups, sp, cell, time_bound, dist_bound };
    let jobs: Vec<usize> = (0..kn - 1).filter(|&k| step[k] && lengths[k + 1] - lengths[k] > 1.0).collect();
    let extra: Vec<Point> = jobs
        .par_iter()
        .flat_map_iter(|&k| {
            let at = |kk: usize| -> Vec<(usize, f32, i32)> {
                who[k].iter().map(|&r| { let (v, t) = grid.at(r, cell, kk); (r, v, t) }).collect()
            };
            cx.refine(lengths[k], lengths[k + 1], &who[k], &at(k), &at(k + 1), env[k + 1])
        })
        .collect();
    if extra.is_empty() {
        return coarse;
    }
    // the exact sequence: grid points, and the best of the evaluated rows at each in-between length
    pts.extend(extra);
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut seq: Vec<Point> = Vec::with_capacity(pts.len());
    for p in pts {
        match seq.last_mut() {
            Some(q) if q.0 == p.0 => {
                if better(p.1, q.1) {
                    *q = p;
                }
            }
            _ => seq.push(p),
        }
    }
    group(by_act, rows, basis, &seq)
}

/// Frontiers of many sets (each a list of rows) in every cell, in parallel.
#[allow(clippy::too_many_arguments)]
pub fn all(
    by_act: &[Vec<&Segment>],
    rows: &[usize],
    grid: &Grid,
    sets: &[Vec<usize>],
    lengths: &[f64],
    metric: usize,
    basis: Basis,
    filters: &[Filter],
    groups: &[Group],
    sp: &Sport,
    refine: bool,
) -> Vec<Effort> {
    let jobs: Vec<(usize, usize)> = (0..sets.len()).flat_map(|s| (0..grid.cells).map(move |c| (s, c))).collect();
    let mut done: Vec<(usize, usize, Vec<Effort>)> = jobs
        .par_iter()
        .map(|&(s, c)| (s, c, one(by_act, rows, grid, &sets[s], c, lengths, metric, basis, filters, groups, sp, refine)))
        .collect();
    done.sort_by_key(|(s, c, _)| (*s, *c));
    let mut out = Vec::new();
    for (s, c, es) in done {
        for mut e in es {
            e.set = s;
            e.cell = c;
            out.push(e);
        }
    }
    out
}
