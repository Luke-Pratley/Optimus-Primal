import optimusprimal.map_uncertainty as map_uncertainty
import pytest
import numpy as np

def test_bisection():
    x_sol = np.zeros((16, 16))
    obj = lambda x: np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, [0, 4], 9 * 16 * 16, 1e3, 1e-3)
    assert np.allclose(val, 3., 1e-2)

    x_sol = np.zeros((128, 128))
    obj = lambda x: np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, [0, 4], 9 * 128 * 128, 1e3, 1e-3)
    assert np.allclose(val, 3., 1e-2)

    x_sol = np.zeros((128, 128))
    obj = lambda x:  np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, [0, 4], 16 * 128 * 128, 1e3, 1e-3)
    assert np.allclose(val, 4., 1e-2)

    x_sol = np.zeros((128, 128))
    obj = lambda x:  np.sum(x**2)
    region = np.ones(x_sol.shape, dtype=bool)
    val = map_uncertainty.bisection_method_credible_interval(x_sol, region, obj, [-4, 0], 16 * 128 * 128, 1e3, 1e-3)
    assert np.allclose(val, -4., 1e-2)

    x_sol = np.zeros((128, 128))
    region_size = 16
    obj = lambda x:  np.sum(np.abs(x)**2)/ (region_size * region_size)
    bound =  4
    iters = 1e2
    tol = 1e-3
    error_p, error_m = map_uncertainty.create_credible_region(x_sol, region_size, obj, bound, iters, tol, 3, -3)
    assert np.allclose(error_p.shape, (128/region_size, 128/region_size), 1e-3)
    assert np.allclose(error_m.shape, (128/region_size, 128/region_size), 1e-3)
    assert np.allclose(error_p, 2., 1e-2)
    assert np.allclose(error_m, -2., 1e-2)
