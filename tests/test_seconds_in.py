"""seconds_in (time spent per bin) against compute and against a count by hand."""

import numpy as np
import pytest

import effort_curves as ec
from test_compute import P  # noqa: F401  (the shared fixture)

M = {m: i for i, m in enumerate(ec.METRICS)}


@pytest.mark.parametrize("dim,edges,filters", [
    ("h", np.arange(120.0, 162.0, 2.0), []),
    ("g", np.arange(-0.2, 0.21, 0.01), [("h", 130.0, 150.0)]),
    ("t0", np.arange(0.0, 4000.0, 120.0), [("g", -0.05, 0.05)]),
    ("v", np.arange(0.0, 7.0, 0.25), [("t0", 300.0, float("nan"))]),
])
def test_a_bin_has_time_exactly_when_it_has_a_window(P, dim, edges, filters):
    s = P.seconds_in(dim, edges, 30.0, filters)
    assert s.shape == (len(P.activity_ids), len(edges) - 1)
    y, _ = P.compute("time", np.array([30.0]), filters=filters, group=(dim, edges))
    # compute's best at 30 s in a bin exists exactly when some 30 s window falls in it
    has = ~np.isnan(y[:, :, 0, M["v"]])
    np.testing.assert_array_equal(s > 0, has)


def test_bins_add_up(P):
    fine = np.arange(100.0, 181.0, 1.0)
    s = P.seconds_in("h", fine, 30.0)
    c = P.seconds_in("h", fine[::4].copy(), 30.0)
    np.testing.assert_array_equal(s.reshape(len(s), -1, 4).sum(axis=2), c)


def test_start_bins_count_every_second(P):
    # time into the activity is defined for every start: one bin over everything counts every full window
    s = P.seconds_in("t0", np.array([-1.0, 1e9]), 30.0)
    s1 = P.seconds_in("t0", np.array([-1.0, 1e9]), 1.0)
    # 29 fewer starts per segment, and at least one segment per activity
    assert np.all(s1[:, 0] - s[:, 0] >= 29)
    assert np.all((s1[:, 0] - s[:, 0]) % 29 == 0)
    # and each start is one second: bins of a second on t0 hold 0 or 1
    e = np.arange(0.0, 5000.0, 1.0)
    one = P.seconds_in("t0", e, 30.0)
    assert set(np.unique(one)) <= {0.0, 1.0}
    sub = P.seconds_in("t0", e, 30.0, activities=np.array([1, 0]))
    np.testing.assert_array_equal(sub, one[[1, 0]])


def test_bad_input(P):
    with pytest.raises(ValueError):
        P.seconds_in("h", np.array([1.0]), 30.0)
    with pytest.raises(ValueError):
        P.seconds_in("h", np.array([1.0, 2.0]), 0.0)
    with pytest.raises(ValueError):
        P.seconds_in("zz", np.array([1.0, 2.0]))
