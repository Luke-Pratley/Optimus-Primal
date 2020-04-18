import optimusprimal.map_uncertainty as map_uncertainty
import pytest
from numpy import linalg as LA
import numpy as np

def test_bisection():
    x_sol = np.zeros((16, 16))
    obj = lambda x: np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, 9 * 16 * 16, 1e3, 1e-3)
    assert np.allclose(val, 3., 1e-2)

    x_sol = np.zeros((128, 128))
    obj = lambda x: np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, 9 * 128 * 128, 1e3, 1e-3)
    assert np.allclose(val, 3., 1e-2)

    x_sol = np.zeros((128, 128))
    obj = lambda x:  np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, 16 * 128 * 128, 1e3, 1e-3)
    assert np.allclose(val, 4., 1e-2)
