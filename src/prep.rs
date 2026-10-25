//! Raw samples -> one 1 Hz grid of running totals per activity.
//!
//! Gaps in the recording (pauses, smart recording) are bridged, never split: distance and altitude are
//! interpolated across them (a real pause records no distance, so pace there is zero by itself), power
//! is zero, and heart rate is interpolated between the readings either side. Every window's numbers
//! are then plain differences of running totals. Glitches are removed per sample (impossible distance
//! steps and climbs count as zero), not per window.

/// Per-sport limits.
#[derive(Clone, Copy, Debug)]
pub struct Sport {
    /// fastest believable speed (m/s): a second with a longer distance step counts as no movement
    pub speed_cap: f64,
    /// steepest believable net gradient of a window (window-level, gradient has no running-total guarantee anyway)
    pub grade_cap: f64,
    /// fastest believable climb (m/s): a second climbing faster counts as no climbing
    pub ascent_cap: f64,
    /// vertical speed only for windows at least this long (s)
    pub ascent_min: f64,
    /// grade-adjusted pace only for windows at least this long (s) / far (m)
    pub gap_min_time: f64,
    pub gap_min_dist: f64,
    /// highest believable cadence (cycles per minute): a second above it counts as no pedaling or stepping
    pub cadence_cap: f64,
}

/// Grid and smoothing parameters.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// activities shorter than this (s) are dropped
    pub min_len: i64,
    /// heart rate is interpolated across stretches without a reading up to this long (s); longer
    /// stretches stay uncovered and fall under the windows' heart-rate coverage rule
    pub hr_bridge: usize,
    /// a sample counts as current for this many seconds after it was recorded
    pub fresh: i64,
    /// altitude smoothing half-width (s)
    pub alt_half: usize,
    /// half-width (s) of the gradient used for grade adjustment
    pub gap_half: usize,
    /// first seconds of an activity left without grade adjustment (barometer settling)
    pub gap_settle: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { min_len: 30, hr_bridge: 600, fresh: 2, alt_half: 15, gap_half: 15, gap_settle: 60.0 }
    }
}

/// One activity on a 1 Hz grid. Index k is second k of the activity;
/// running totals include second k, so a window (i, j] sums seconds i+1..=j.
pub struct Segment {
    /// index of the activity this segment belongs to
    pub act: usize,
    /// activity seconds at index 0 (time into the activity)
    pub t_off: f64,
    /// recorded distance into the activity at index 0 (m)
    pub d_off: f64,
    /// clean cumulative distance (m)
    pub cd: Vec<f64>,
    /// cumulative work (J); empty when the activity has no power
    pub cp: Vec<f64>,
    /// cumulative heart beats-seconds and count of seconds with a valid heart rate
    pub ch: Vec<f64>,
    pub chn: Vec<f64>,
    /// smoothed altitude (NaN where unknown)
    pub ca: Vec<f64>,
    /// cumulative gross ascent of the smoothed altitude (m)
    pub cg: Vec<f64>,
    /// cumulative grade-adjusted distance (m); empty without a model
    pub cx: Vec<f64>,
    /// cumulative cycles (pedal revolutions or steps) and seconds spent pedaling or stepping (cadence > 0);
    /// empty when the activity recorded no cadence
    pub crev: Vec<f64>,
    pub cped: Vec<f64>,
    pub has_alt: bool,
}

impl Segment {
    #[inline]
    pub fn n(&self) -> usize {
        self.cd.len()
    }
}

#[inline]
fn cumsum(x: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    let mut s = 0.0;
    for &v in x {
        s += v;
        out.push(s);
    }
    out
}

/// Build one activity's running totals (one segment, or none if it is too short).
///
/// `t` is seconds into the activity, sorted and unique. `gap` is the grade-adjustment factor per whole
/// percent of gradient from -35 to +35 (71 values), or None.
#[allow(clippy::too_many_arguments)]
pub fn prepare_activity(
    act: usize,
    t: &[i64],
    dist: &[f64],
    alt: &[f64],
    pw: &[f64],
    hr: &[f64],
    cad: &[f64],
    indoor: bool,
    powered: bool,
    gap: Option<&[f64]>,
    sp: &Sport,
    pa: &Params,
) -> Vec<Segment> {
    if t.is_empty() || t[t.len() - 1] - t[0] < pa.min_len {
        return Vec::new();
    }
    vec![build_segment(act, t, dist, alt, pw, hr, cad, indoor, powered, gap, sp, pa)]
}

#[allow(clippy::too_many_arguments)]
fn build_segment(
    act: usize,
    t: &[i64],
    dist: &[f64],
    alt: &[f64],
    pw: &[f64],
    hr: &[f64],
    cad: &[f64],
    indoor: bool,
    powered: bool,
    gap: Option<&[f64]>,
    sp: &Sport,
    pa: &Params,
) -> Segment {
    let t0 = t[0];
    let len = (t[t.len() - 1] - t0) as usize + 1;
    let mut d = vec![f64::NAN; len];
    let mut ar = vec![f64::NAN; len];
    let mut p = vec![0.0; if powered { len } else { 0 }];
    let mut h = vec![0.0; len];
    let mut hn = vec![0.0; len];
    // cadence in cycles per minute: one second adds cadence / 60 cycles; gaps and glitches add none
    let mut cy = vec![0.0; len];
    let mut cs = vec![0.0; len];

    // onto the grid: distance and altitude interpolated across gaps (the elapsed time counts, the
    // movement is spread over it); power and heart rate only from current samples
    let mut ptr = 0;
    let mut last_d = dist[0];
    for g in 0..len {
        let tg = t0 + g as i64;
        while ptr + 1 < t.len() && t[ptr + 1] <= tg {
            ptr += 1;
            if !dist[ptr].is_nan() {
                last_d = dist[ptr];
            }
        }
        let fresh = tg - t[ptr] <= pa.fresh;
        let nx = ptr + 1;
        let frac = if nx < t.len() { (tg - t[ptr]) as f64 / (t[nx] - t[ptr]) as f64 } else { 0.0 };
        d[g] = if frac > 0.0 && !dist[ptr].is_nan() && !dist[nx].is_nan() {
            dist[ptr] + (dist[nx] - dist[ptr]) * frac
        } else {
            last_d
        };
        if !indoor {
            ar[g] = if frac > 0.0 && !alt[ptr].is_nan() && !alt[nx].is_nan() {
                alt[ptr] + (alt[nx] - alt[ptr]) * frac
            } else if fresh {
                alt[ptr]
            } else {
                f64::NAN
            };
        }
        if fresh {
            if powered && !pw[ptr].is_nan() {
                p[g] = pw[ptr];
            }
            let c = cad[ptr];
            if c > 0.0 && c <= sp.cadence_cap {
                cy[g] = c / 60.0;
                cs[g] = 1.0;
            }
            let v = hr[ptr];
            if (30.0..=230.0).contains(&v) {
                h[g] = v;
                hn[g] = 1.0;
            }
        }
    }

    // heart rate: interpolate across stretches without a reading (pauses, smart recording, short dropouts)
    let mut last: Option<usize> = None;
    for g in 0..len {
        if hn[g] == 0.0 {
            continue;
        }
        if let Some(l) = last {
            if g - l > 1 && g - l <= pa.hr_bridge {
                for k in l + 1..g {
                    h[k] = h[l] + (h[g] - h[l]) * (k - l) as f64 / (g - l) as f64;
                    hn[k] = 1.0;
                }
            }
        }
        last = Some(g);
    }

    // distance: per-second steps; a second faster than the speed cap is a GPS jump (or a vehicle) and
    // counts as no movement, so a run of them can never add up to a fast window
    let mut step = vec![0.0; len];
    for g in 1..len {
        if !d[g].is_nan() && !d[g - 1].is_nan() {
            let s = d[g] - d[g - 1];
            if s > 0.0 && s <= sp.speed_cap {
                step[g] = s;
            }
        }
    }
    let cd = cumsum(&step);
    let d_off = d.iter().copied().find(|v| !v.is_nan()).unwrap_or(0.0);

    // altitude: centred mean of real samples, then gross ascent (impossible climbs dropped)
    let mut ps = vec![0.0; len + 1];
    let mut pc = vec![0.0; len + 1];
    for g in 0..len {
        let ok = !ar[g].is_nan();
        ps[g + 1] = ps[g] + if ok { ar[g] } else { 0.0 };
        pc[g + 1] = pc[g] + if ok { 1.0 } else { 0.0 };
    }
    let mut ca = vec![f64::NAN; len];
    let hw = pa.alt_half;
    for g in 0..len {
        let lo = g.saturating_sub(hw);
        let hi = (g + hw).min(len - 1) + 1;
        let c = pc[hi] - pc[lo];
        if c > 0.0 {
            ca[g] = (ps[hi] - ps[lo]) / c;
        }
    }
    let has_alt = ca.iter().any(|v| !v.is_nan());
    let mut up = vec![0.0; len];
    for g in 1..len {
        if !ca[g].is_nan() && !ca[g - 1].is_nan() {
            let u = ca[g] - ca[g - 1];
            // faster than the ascent cap is a barometer jump: no climbing
            if u > 0.0 && u <= sp.ascent_cap {
                up[g] = u;
            }
        }
    }
    let cg = cumsum(&up);

    // grade-adjusted distance: each second's step divided by the relative speed on its gradient
    let cx = match gap {
        Some(f) => {
            let k = pa.gap_half;
            let mut x = vec![0.0; len];
            for g in 0..len {
                let mut fac = 1.0;
                if (t0 + g as i64) as f64 >= pa.gap_settle && g >= k && g + k < len {
                    let span = cd[g + k] - cd[g - k];
                    let da = ca[g + k] - ca[g - k];
                    if span > 10.0 && !da.is_nan() {
                        let gr = da / span;
                        if gr.abs() <= 0.45 {
                            let idx = ((gr * 100.0).round().clamp(-35.0, 35.0) + 35.0) as usize;
                            fac = f[idx];
                        }
                    }
                }
                x[g] = step[g] / fac;
            }
            cumsum(&x)
        }
        None => Vec::new(),
    };

    Segment {
        act,
        t_off: t0 as f64,
        d_off,
        cd,
        cp: if powered { cumsum(&p) } else { Vec::new() },
        ch: cumsum(&h),
        crev: if cs.iter().any(|&x| x > 0.0) { cumsum(&cy) } else { Vec::new() },
        cped: if cs.iter().any(|&x| x > 0.0) { cumsum(&cs) } else { Vec::new() },
        chn: cumsum(&hn),
        ca,
        cg,
        cx,
        has_alt,
    }
}
