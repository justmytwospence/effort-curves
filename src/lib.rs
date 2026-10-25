//! Python bindings for the effort-curve prototype.

pub mod compute;
pub mod prep;

use compute::{Basis, Dim, Filter, Group, M, METRICS};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
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

    /// Best of each metric per activity and length.
    ///
    /// Returns (y, t0): float32 and int32 arrays of shape (activities, lengths, metrics), or
    /// (activities, groups, lengths, metrics) with `group`; metrics in the order p, v, g, h, a, x.
    /// filters: (dim, lo, hi) with dim in p v g h a x t0 d0 and NaN for an open end (lo <= value <= hi).
    /// group: (dim, edges); a window falls into bin i when edges[i] <= value < edges[i + 1], else it is skipped.
    /// activities: indices into activity_ids to evaluate (default all), in output order.
    #[pyo3(signature = (basis, lengths, eps=0.0, filters=vec![], group=None, activities=None))]
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
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let basis = match basis {
            "time" => Basis::Time,
            "dist" => Basis::Dist,
            b => return Err(PyValueError::new_err(format!("unknown basis {b}"))),
        };
        let lengths = lengths.as_slice()?.to_vec();
        let filters = parse_filters(filters)?;
        let group = match group {
            Some((d, e)) => {
                let edges = e.as_slice()?.to_vec();
                if edges.len() < 2 || edges.windows(2).any(|w| !(w[0] < w[1])) {
                    return Err(PyValueError::new_err("group edges must be at least two increasing values"));
                }
                Some(Group { dim: parse_dim(&d)?, edges })
            }
            None => None,
        };
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
            compute::all(&self.by_act(), &acts, basis, &lengths, eps, &filters, group.as_ref(), &self.sport)
        });
        let (a, k) = (acts.len(), lengths.len());
        let g = group.as_ref().map(|g| g.n());
        let mut y = Vec::with_capacity(a * k * M);
        let mut t0 = Vec::with_capacity(a * k * M);
        for r in results {
            y.extend_from_slice(&r.y);
            t0.extend_from_slice(&r.t0);
        }
        let (y, t0) = match g {
            Some(g) => (
                PyArray1::from_vec(py, y).reshape([a, g, k, M])?.into_any(),
                PyArray1::from_vec(py, t0).reshape([a, g, k, M])?.into_any(),
            ),
            None => (
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
