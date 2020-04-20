import numpy as np


def bisection_method_credible_interval(objective_function, start_interval, iters, tol):
    """
    Bisection method for finding credible interval
    """

    eta1 = start_interval[0]
    eta2 = start_interval[1]
    obj3 = objective_function(eta2)
    if(np.sign(objective_function(eta1)) == np.sign(objective_function(eta2))):
        print("eta1 and eta2 have same sign.")
    for i in range(int(iters)):
        obj1 = objective_function(eta1)
        eta3 = (eta2 + eta1) * 0.5
        obj3 = objective_function(eta3)
        if(np.abs(obj3) < tol or np.abs(obj1) < tol):
            return eta3
        if(np.abs(eta1 - eta2) < 1e-12):
            return eta3
        if(np.sign(obj1) == np.sign(obj3)):
            eta1 = eta3
        else:
            eta2 = eta3
    print("Did not converge... ",obj3)
    return eta3


def create_credible_region(x_sol, region_size, objective_function, bound, iters, tol, top):
    """
    Bisection method for finding credible interval
    """

    region = np.zeros(x_sol.shape)
    if len(x_sol.shape) > 1:
        region[:region_size, :region_size] = 1.
        dsizey, dsizex = int(x_sol.shape[0]/region_size), int(x_sol.shape[1]/region_size)
        error_p = np.zeros((dsizey, dsizex))
        error_m = np.zeros((dsizey, dsizex))
        mean = np.zeros((dsizey, dsizex))
        for i in range(dsizey):
            for j in range(dsizex):
                mask = np.roll(np.roll(region, shift=i * region_size, axis=0), shift=j * region_size, axis=1)
                x_mean = np.mean(np.ravel(x_sol[(mask.astype(bool))]))
                mean[i, j] = x_mean
                obj = lambda eta : objective_function(x_sol * (1. - mask) + eta * mask) - bound
                error_p[i, j] = bisection_method_credible_interval(obj, [0, top], iters, tol)
                obj = lambda eta : objective_function(x_sol * (1. - mask) - eta * mask ) - bound
                error_m[i, j] = -bisection_method_credible_interval(obj, [0, top], iters, tol)
                print(i, j, (error_p[i, j], error_m[i, j]), x_mean)
    else:
        region[:region_size] = 1.
        dsizey = int(x_sol.shape[0]/region_size)
        error_p = np.zeros((dsizey))
        error_m = np.zeros((dsizey))
        mean = np.zeros((dsizey))
        for i in range(dsizey):
                mask = np.roll(region, shift=i * region_size, axis=0)
                x_mean = np.mean(np.ravel(x_sol[(mask.astype(bool))]))
                mean[i] = x_mean
                obj = lambda eta : objective_function(x_sol * (1. - mask) + eta * mask) - bound
                error_p[i] = bisection_method_credible_interval(obj, [0, top], iters, tol)
                obj = lambda eta : objective_function(x_sol * (1. - mask) - eta * mask) - bound
                error_m[i] = -bisection_method_credible_interval(obj, [0, top], iters, tol)
                print(i, (error_p[i], error_m[i]), x_mean)
    return error_p, error_m, mean
