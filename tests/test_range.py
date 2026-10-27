"""compute(metrics=) and range_frames against numpy."""

import numpy as np
import pytest

import effort_curves as ec
from test_compute import P  # noqa: F401  (the shared fixture)

M = {m: i for i, m in enumerate(ec.METRICS)}


def test_metrics_subset_matches_full(P):
    L = np.array([30.0, 300.0])
    edges = np.array([-1.0, 0.0, 0.03, 1.0])
    y, t = P.compute("time", L, group=("g", edges))
    ys, ts = P.compute("time", L, group=("g", edges), metrics=["v", "h"])
    assert ys.shape == y.shape[:-1] + (2,)
    np.testing.assert_array_equal(ys[..., 0], y[..., M["v"]])
    np.testing.assert_array_equal(ts[..., 1], t[..., M["h"]])
    with pytest.raises(ValueError):
        P.compute("time", L, metrics=["zz"])


@pytest.mark.parametrize("width", [1, 3, 7])
def test_range_frames_is_a_sliding_max(width):
    rng = np.random.default_rng(width)
    y = rng.normal(size=(4, 12, 5)).astype(np.float32)
    y[rng.random(y.shape) < 0.3] = np.nan
    t = rng.integers(0, 1000, size=y.shape).astype(np.int32)
    oy, ot = ec.range_frames(y, t, width)
    assert oy.shape == (4, 12 - width + 1, 5)
    for p in range(12 - width + 1):
        win = y[:, p:p + width, :]
        ref = np.where(np.all(np.isnan(win), axis=1), np.nan, np.nanmax(np.where(np.isnan(win), -np.inf, win), axis=1))
        np.testing.assert_array_equal(oy[:, p, :], ref.astype(np.float32))
        # the t0 is that of a bin in the range that holds the maximum
        for r in range(4):
            for c in range(5):
                if np.isnan(ref[r, c]):
                    assert ot[r, p, c] == -1
                else:
                    hold = [b for b in range(p, p + width) if y[r, b, c] == ref[r, c]]
                    assert ot[r, p, c] in [t[r, b, c] for b in hold]


def test_range_frames_bad_width():
    y = np.zeros((1, 3, 1), np.float32)
    with pytest.raises(ValueError):
        ec.range_frames(y, y.astype(np.int32), 0)
    with pytest.raises(ValueError):
        ec.range_frames(y, y.astype(np.int32), 4)
