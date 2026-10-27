//! Python bindings for the effort-curve prototype.

pub mod compute;
pub mod frontier;
pub mod into;
pub mod prep;

use compute::{Basis, Dim, Filter, Group, M, METRICS};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::types::PyDict;
use prep::{Params, Segment, Sport};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Prepared segments for a set of activities, ready for any number of window evaluations.
#[pyclass(frozen)]
struct Prepared {
    activity_ids: Vec<i64>,
    segs: Vec<Segment>,
    sport: Sport,
}

impl Prepared {
    fn by_act(&self) -> Vec<Vec<&Segment>> {
        let mut out: Vec<Vec<&Segment>> = vec![Vec::new(); self.activity_ids.len()];
        for s in &self.segs {
            out[s.act].push(s);
        }
        out
    }
}

fn parse_dim(d: &str) -> PyResult<Dim> {
    Ok(match d {
        "t0" => Dim::T0,
        "d0" => Dim::D0,
        m => Dim::Metric(METRICS.iter().position(|x| *x == m).ok_or_else(|| PyValueError::new_err(format!("unknown dim {m}")))?),
    })
}

fn parse_filters(filters: Vec<(String, f64, f64)>) -> PyResult<Vec<Filter>> {
    filters.into_iter().map(|(d, lo, hi)| Ok(Filter { dim: parse_dim(&d)?, lo, hi })).collect()
}

fn parse_groups<'py>(
    group: Option<(String, PyReadonlyArray1<'py, f64>)>,
    group2: Option<(String, PyReadonlyArray1<'py, f64>)>,
) -> PyResult<Vec<Group>> {
    if group.is_none() && group2.is_some() {
        return Err(PyValueError::new_err("group2 needs group"));
    }
    let mut groups = Vec::new();
    for (d, e) in group.into_iter().chain(group2) {
        let edges = e.as_slice()?.to_vec();
        if edges.len() < 2 || edges.windows(2).any(|w| !(w[0] < w[1])) {
            return Err(PyValueError::new_err("group edges must be at least two increasing values"));
        }
        groups.push(Group { dim: parse_dim(&d)?, edges });
    }
    Ok(groups)
}

fn parse_basis(basis: &str) -> PyResult<Basis> {
    match basis {
        "time" => Ok(Basis::Time),
        "dist" => Ok(Basis::Dist),
        b => Err(PyValueError::new_err(format!("unknown basis {b}"))),
    }
}

#[pymethods]
impl Prepared {
    #[getter]
    fn activity_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.activity_ids)
    }

    #[getter]
    fn n_segments(&self) -> usize {
        self.segs.len()
    }

    /// Seconds on the 1 Hz grid, across all segments.
    #[getter]
    fn seconds(&self) -> usize {
        self.segs.iter().map(|s| s.n()).sum()
    }

    /// The frontier of one metric as efforts, for each set of activities and each group cell.
    ///
    /// y, t0: `compute`'s output for that metric (not enveloped), shaped (rows, cells, lengths), cells being the
    /// product of the groups (1 without). activities: the Prepared index of each row (as passed to `compute`).
    /// sets: lists of rows, one frontier each (a single row gives that activity's own efforts). filters, group,
    /// group2: as given to `compute`, so refined windows obey them too. refine: find each handoff to the second or
    /// metre (else at grid resolution).
    /// Returns a dict of arrays, one entry per effort: set, cell, effort (order along the frontier), row, t0 (start of
    /// its full window), len (its full length), y (its value there), from_len (it holds the frontier above this
    /// length; NaN for the first), from_y (the frontier just above from_len).
    #[pyo3(signature = (basis, metric, lengths, y, t0, activities, sets, filters=vec![], group=None, group2=None, refine=true))]
    #[allow(clippy::too_many_arguments)]
    fn frontier<'py>(
        &self,
        py: Python<'py>,
        basis: &str,
        metric: &str,
        lengths: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray3<'py, f32>,
        t0: PyReadonlyArray3<'py, i32>,
        activities: PyReadonlyArray1<'py, i64>,
        sets: Vec<Vec<usize>>,
        filters: Vec<(String, f64, f64)>,
        group: Option<(String, PyReadonlyArray1<'py, f64>)>,
        group2: Option<(String, PyReadonlyArray1<'py, f64>)>,
        refine: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let basis = parse_basis(basis)?;
        let m = METRICS.iter().position(|x| *x == metric).ok_or_else(|| PyValueError::new_err(format!("unknown metric {metric}")))?;
        let lengths = lengths.as_slice()?.to_vec();
        if lengths.windows(2).any(|w| !(w[0] < w[1])) {
            return Err(PyValueError::new_err("lengths must increase"));
        }
        let filters = parse_filters(filters)?;
        let groups = parse_groups(group, group2)?;
        let cells: usize = groups.iter().map(|g| g.n()).product();
        let shape = y.shape().to_vec();
        if shape != t0.shape() || shape[1] != cells || shape[2] != lengths.len() {
            return Err(PyValueError::new_err(format!("y and t0 must be shaped (rows, {cells}, {})", lengths.len())));
        }
        let n_all = self.activity_ids.len();
        let rows: Vec<usize> = activities
            .as_slice()?
            .iter()
            .map(|&i| if i >= 0 && (i as usize) < n_all { Ok(i as usize) } else { Err(PyValueError::new_err(format!("activity index {i} out of range"))) })
            .collect::<PyResult<_>>()?;
        if rows.len() != shape[0] {
            return Err(PyValueError::new_err("one activity per row of y"));
        }
        if sets.iter().flatten().any(|&r| r >= rows.len()) {
            return Err(PyValueError::new_err("set refers to a row out of range"));
        }
        let (ys, ts) = (y.as_slice()?, t0.as_slice()?);
        let grid = frontier::Grid { y: ys, t0: ts, cells, k: lengths.len() };
        let by_act = self.by_act();
        let eff = py.allow_threads(|| frontier::all(&by_act, &rows, &grid, &sets, &lengths, m, basis, &filters, &groups, &self.sport, refine));
        let d = PyDict::new(py);
        d.set_item("set", PyArray1::from_vec(py, eff.iter().map(|e| e.set as i64).collect()))?;
        d.set_item("cell", PyArray1::from_vec(py, eff.iter().map(|e| e.cell as i64).collect()))?;
        d.set_item("effort", PyArray1::from_vec(py, eff.iter().map(|e| e.effort as i64).collect()))?;
        d.set_item("row", PyArray1::from_vec(py, eff.iter().map(|e| e.row as i64).collect()))?;
        d.set_item("t0", PyArray1::from_vec(py, eff.iter().map(|e| e.t0).collect()))?;
        d.set_item("len", PyArray1::from_vec(py, eff.iter().map(|e| e.len).collect()))?;
        d.set_item("y", PyArray1::from_vec(py, eff.iter().map(|e| e.y).collect()))?;
        d.set_item("from_len", PyArray1::from_vec(py, eff.iter().map(|e| e.from_len).collect()))?;
        d.set_item("from_y", PyArray1::from_vec(py, eff.iter().map(|e| e.from_y).collect()))?;
        Ok(d)
    }

    /// The frontier over where a window starts: for windows of `length` (on `basis`), the best of `metric` among those
    /// starting at least x into an activity, x being seconds (`dim` "t0") or metres ("d0") in, for each set of rows.
    /// Exact: every start of every activity counts (windows passing `filters`). activities: the Prepared index of each
    /// row. Returns (efforts, points). efforts: set, effort, row, t0 (start of its last staircase window, s into the
    /// activity), x, y (that window's), from_x (it holds the frontier above this; NaN for the first), from_y (the
    /// frontier just above from_x). points: the staircase (set, effort, x, y), ascending x within a set.
    #[pyo3(signature = (basis, length, metric, dim, activities, sets, filters=vec![]))]
    #[allow(clippy::too_many_arguments)]
    fn frontier_into<'py>(
        &self,
        py: Python<'py>,
        basis: &str,
        length: f64,
        metric: &str,
        dim: &str,
        activities: PyReadonlyArray1<'py, i64>,
        sets: Vec<Vec<usize>>,
        filters: Vec<(String, f64, f64)>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let basis = parse_basis(basis)?;
        let m = METRICS.iter().position(|x| *x == metric).ok_or_else(|| PyValueError::new_err(format!("unknown metric {metric}")))?;
        let dim = match dim {
            "t0" => Dim::T0,
            "d0" => Dim::D0,
            d => return Err(PyValueError::new_err(format!("dim must be t0 or d0, not {d}"))),
        };
        let filters = parse_filters(filters)?;
        let n_all = self.activity_ids.len();
        let rows: Vec<usize> = activities
            .as_slice()?
            .iter()
            .map(|&i| if i >= 0 && (i as usize) < n_all { Ok(i as usize) } else { Err(PyValueError::new_err(format!("activity index {i} out of range"))) })
            .collect::<PyResult<_>>()?;
        if sets.iter().flatten().any(|&r| r >= rows.len()) {
            return Err(PyValueError::new_err("set refers to a row out of range"));
        }
        let by_act = self.by_act();
        let (eff, pts) = py.allow_threads(|| into::all(&by_act, &rows, &sets, basis, length, m, dim, &filters, &self.sport));
        let e = PyDict::new(py);
        e.set_item("set", PyArray1::from_vec(py, eff.iter().map(|v| v.set as i64).collect()))?;
        e.set_item("effort", PyArray1::from_vec(py, eff.iter().map(|v| v.effort as i64).collect()))?;
        e.set_item("row", PyArray1::from_vec(py, eff.iter().map(|v| v.row as i64).collect()))?;
        e.set_item("t0", PyArray1::from_vec(py, eff.iter().map(|v| v.t0).collect()))?;
        e.set_item("x", PyArray1::from_vec(py, eff.iter().map(|v| v.x).collect()))?;
        e.set_item("y", PyArray1::from_vec(py, eff.iter().map(|v| v.y).collect()))?;
        e.set_item("from_x", PyArray1::from_vec(py, eff.iter().map(|v| v.from_x).collect()))?;
        e.set_item("from_y", PyArray1::from_vec(py, eff.iter().map(|v| v.from_y).collect()))?;
        let p = PyDict::new(py);
        p.set_item("set", PyArray1::from_vec(py, pts.iter().map(|v| v.0 as i64).collect()))?;
        p.set_item("effort", PyArray1::from_vec(py, pts.iter().map(|v| v.1 as i64).collect()))?;
        p.set_item("x", PyArray1::from_vec(py, pts.iter().map(|v| v.2).collect()))?;
        p.set_item("y", PyArray1::from_vec(py, pts.iter().map(|v| v.3).collect()))?;
        Ok((e, p))
    }

    /// Best of each metric per activity and length.
    ///
    /// Returns (y, t0): float32 and int32 arrays of shape (activities, lengths, metrics), or
    /// (activities, groups, lengths, metrics) with `group`, or (activities, groups, groups2, lengths, metrics) with
    /// `group` and `group2`; metrics in the order of METRICS.
    /// filters: (dim, lo, hi) with dim in p v g h a x t0 d0 and NaN for an open end (lo <= value <= hi).
    /// group: (dim, edges); a window falls into bin i when edges[i] <= value < edges[i + 1], else it is skipped.
    /// group2: a second (dim, edges), binned at the same time: one pass gives every combination of the two.
    /// activities: indices into activity_ids to evaluate (default all), in output order.
    #[pyo3(signature = (basis, lengths, eps=0.0, filters=vec![], group=None, activities=None, group2=None))]
    #[allow(clippy::too_many_arguments)]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        basis: &str,
        lengths: PyReadonlyArray1<'py, f64>,
        eps: f64,
        filters: Vec<(String, f64, f64)>,
        group: Option<(String, PyReadonlyArray1<'py, f64>)>,
        activities: Option<PyReadonlyArray1<'py, i64>>,
        group2: Option<(String, PyReadonlyArray1<'py, f64>)>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let basis = parse_basis(basis)?;
        let lengths = lengths.as_slice()?.to_vec();
        let filters = parse_filters(filters)?;
        let groups = parse_groups(group, group2)?;
        let n_all = self.activity_ids.len();
        let acts: Vec<usize> = match activities {
            Some(ix) => ix
                .as_slice()?
                .iter()
                .map(|&i| {
                    if i >= 0 && (i as usize) < n_all {
                        Ok(i as usize)
                    } else {
                        Err(PyValueError::new_err(format!("activity index {i} out of range")))
                    }
                })
                .collect::<PyResult<_>>()?,
            None => (0..n_all).collect(),
        };
        let results = py.allow_threads(|| {
            compute::all(&self.by_act(), &acts, basis, &lengths, eps, &filters, &groups, &self.sport)
        });
        let (a, k) = (acts.len(), lengths.len());
        let dims: Vec<usize> = groups.iter().map(|g| g.n()).collect();
        let mut y = Vec::with_capacity(a * k * M);
        let mut t0 = Vec::with_capacity(a * k * M);
        for r in results {
            y.extend_from_slice(&r.y);
            t0.extend_from_slice(&r.t0);
        }
        let (y, t0) = match dims[..] {
            [g1, g2] => (
                PyArray1::from_vec(py, y).reshape([a, g1, g2, k, M])?.into_any(),
                PyArray1::from_vec(py, t0).reshape([a, g1, g2, k, M])?.into_any(),
            ),
            [g] => (
                PyArray1::from_vec(py, y).reshape([a, g, k, M])?.into_any(),
                PyArray1::from_vec(py, t0).reshape([a, g, k, M])?.into_any(),
            ),
            _ => (
                PyArray1::from_vec(py, y).reshape([a, k, M])?.into_any(),
                PyArray1::from_vec(py, t0).reshape([a, k, M])?.into_any(),
            ),
        };
        Ok((y, t0))
    }
}

/// Build segments from raw samples of many activities, concatenated and sorted by (activity, t).
///
/// offsets has len(activity_ids) + 1 entries delimiting each activity's samples. t is seconds into the
/// activity (unique per activity). NaN marks a missing value. gap: 71 grade-adjustment factors for
/// -35..+35 % gradient, or None. sport: (speed cap, grade cap, ascent cap, ascent min length, GAP min time,
/// GAP min distance, cadence cap). cadence: cycles per minute per sample (pedal revolutions or steps). hr_bridge: longest stretch without a heart-rate reading (s) that is
/// interpolated.
#[pyfunction]
#[pyo3(signature = (activity_ids, offsets, t, dist, alt, power, hr, indoor, powered, sport, gap=None, hr_bridge=600, cadence=None))]
#[allow(clippy::too_many_arguments)]
fn prepare<'py>(
    py: Python<'py>,
    activity_ids: PyReadonlyArray1<'py, i64>,
    offsets: PyReadonlyArray1<'py, i64>,
    t: PyReadonlyArray1<'py, i64>,
    dist: PyReadonlyArray1<'py, f64>,
    alt: PyReadonlyArray1<'py, f64>,
    power: PyReadonlyArray1<'py, f64>,
    hr: PyReadonlyArray1<'py, f64>,
    indoor: PyReadonlyArray1<'py, u8>,
    powered: PyReadonlyArray1<'py, u8>,
    sport: (f64, f64, f64, f64, f64, f64, f64),
    gap: Option<PyReadonlyArray1<'py, f64>>,
    hr_bridge: usize,
    cadence: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Prepared> {
    let ids = activity_ids.as_slice()?.to_vec();
    let off = offsets.as_slice()?;
    if off.len() != ids.len() + 1 {
        return Err(PyValueError::new_err("offsets must have len(activity_ids) + 1 entries"));
    }
    let (t, dist, alt, pw, hr) = (t.as_slice()?, dist.as_slice()?, alt.as_slice()?, power.as_slice()?, hr.as_slice()?);
    let (indoor, powered) = (indoor.as_slice()?, powered.as_slice()?);
    // cadence in cycles per minute (pedal revolutions, or steps); absent means none recorded
    let no_cad;
    let cad: &[f64] = match &cadence {
        Some(c) => c.as_slice()?,
        None => {
            no_cad = vec![f64::NAN; t.len()];
            &no_cad
        }
    };
    if cad.len() != t.len() {
        return Err(PyValueError::new_err("cadence must have one value per sample"));
    }
    let gap = match &gap {
        Some(g) => {
            let g = g.as_slice()?;
            if g.len() != 71 {
                return Err(PyValueError::new_err("gap must have 71 factors (-35..+35 %)"));
            }
            Some(g.to_vec())
        }
        None => None,
    };
    let sport = Sport {
        speed_cap: sport.0,
        grade_cap: sport.1,
        ascent_cap: sport.2,
        ascent_min: sport.3,
        gap_min_time: sport.4,
        gap_min_dist: sport.5,
        cadence_cap: sport.6,
    };
    let pa = Params { hr_bridge, ..Params::default() };
    let segs: Vec<Segment> = py.allow_threads(|| {
        (0..ids.len())
            .into_par_iter()
            .flat_map_iter(|a| {
                let (lo, hi) = (off[a] as usize, off[a + 1] as usize);
                if hi <= lo {
                    return Vec::new();
                }
                prep::prepare_activity(
                    a,
                    &t[lo..hi],
                    &dist[lo..hi],
                    &alt[lo..hi],
                    &pw[lo..hi],
                    &hr[lo..hi],
                    &cad[lo..hi],
                    indoor[a] != 0,
                    powered[a] != 0,
                    gap.as_deref(),
                    &sport,
                    &pa,
                )
            })
            .collect()
    });
    Ok(Prepared { activity_ids: ids, segs, sport })
}

#[pymodule]
fn effort_curves(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prepare, m)?)?;
    m.add_class::<Prepared>()?;
    m.add("METRICS", METRICS.to_vec())?;
    Ok(())
}
