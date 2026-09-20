import numpy as np
from truncation_tell.synthetic import make_pool, truncate_select


def test_pool_shape_and_determinism():
    a = make_pool(n=500, k=8, seed=3)
    b = make_pool(n=500, k=8, seed=3)
    assert a.shape == (500, 8)
    np.testing.assert_allclose(a, b)


def test_pool_spectrum_is_power_law_not_isotropic():
    """The null must already be low-rank, or the detector proves nothing."""
    pool = make_pool(n=20000, k=16, alpha=1.0, seed=0)
    eigs = np.linalg.eigvalsh(np.cov(pool, rowvar=False))[::-1]
    # Leading direction carries far more variance than the trailing one.
    assert eigs[0] / eigs[-1] > 8.0
    # Decay follows a power law: the log-log slope recovers -alpha.
    slope = np.polyfit(np.log(np.arange(1, 17)), np.log(eigs), 1)[0]
    assert -1.35 < slope < -0.65


def test_pool_is_symmetric_along_every_axis():
    """Unselected data is near-symmetric; this is what truncation breaks."""
    from scipy.stats import skew
    pool = make_pool(n=40000, k=8, seed=1)
    assert np.max(np.abs(skew(pool, axis=0))) < 0.1


def test_truncate_select_returns_top_quantile():
    pool = make_pool(n=1000, k=8, seed=5)
    direction = np.zeros(8)
    direction[2] = 1.0
    idx = truncate_select(pool, direction, gamma=0.1)
    assert idx.shape == (100,)
    selected = pool[idx, 2]
    remaining = np.delete(pool[:, 2], idx)
    assert selected.min() >= remaining.max()


def test_truncate_select_induces_skew_along_direction():
    """The signature under test: truncation manufactures asymmetry."""
    from scipy.stats import skew
    pool = make_pool(n=20000, k=8, seed=6)
    direction = np.ones(8) / np.sqrt(8)
    idx = truncate_select(pool, direction, gamma=0.1)
    proj_pool = pool @ direction
    proj_sel = pool[idx] @ direction
    assert abs(skew(proj_pool)) < 0.1
    assert skew(proj_sel) > 0.5
    # And truncation deflates variance along that direction.
    assert proj_sel.var() < 0.5 * proj_pool.var()


def test_truncate_select_is_direction_normalized():
    pool = make_pool(n=1000, k=8, seed=7)
    d = np.ones(8)
    a = truncate_select(pool, d, gamma=0.1)
    b = truncate_select(pool, 17.0 * d, gamma=0.1)
    np.testing.assert_array_equal(np.sort(a), np.sort(b))
