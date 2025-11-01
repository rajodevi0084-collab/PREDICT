import numpy as np

from src.postproc.tick_snap import clip_log_return, snap_to_tick


def test_snap_to_tick_scalar_and_array():
    assert np.isclose(snap_to_tick(101.03, 0.05), 101.05)
    arr = np.array([101.01, 101.07])
    snapped = snap_to_tick(arr, 0.05)
    assert np.allclose(snapped, [101.0, 101.05])


def test_clip_log_return():
    values = np.array([0.1, -0.2, 0.05])
    clipped = clip_log_return(values, k_sigma=2.0, sigma_train=0.05)
    assert np.allclose(clipped, [0.1, -0.1, 0.05])
