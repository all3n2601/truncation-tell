import numpy as np
import pytest
from truncation_tell.attack import lls_select


def test_lls_select_keeps_only_positive_weights():
    weights = np.array([-3.0, -1.0, 0.0, 2.0, 5.0, 7.0])
    idx = lls_select(weights, gamma=1.0)
    assert set(idx) == {3, 4, 5}


def test_lls_select_takes_the_top_gamma_fraction_of_positives():
    weights = np.arange(-5.0, 5.0)          # four strictly positive entries
    idx = lls_select(weights, gamma=0.4)    # round(0.4 * 4) = 2
    assert len(idx) == 2
    assert set(idx) == {8, 9}


def test_lls_select_raises_when_nothing_is_positive():
    with pytest.raises(ValueError):
        lls_select(np.array([-1.0, -2.0, 0.0]), gamma=0.5)


def test_lls_select_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        lls_select(np.array([1.0, 2.0]), gamma=0.0)
    with pytest.raises(ValueError):
        lls_select(np.array([1.0, 2.0]), gamma=1.5)


def test_lls_select_returns_at_least_one_index():
    weights = np.array([-1.0, 0.5, 1.0])
    idx = lls_select(weights, gamma=0.01)
    assert len(idx) == 1
    assert idx[0] == 2
