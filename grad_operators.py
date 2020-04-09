import numpy as np
import operators

class L2_Grad:
    """
    This class computes the gradiant operator of the l2 norm function

                        f(x) = ||y - Wx||^2/2/sigma^2

    When the input 'x' is an array. 'y' is a data vector, `sigma` is a scalar uncertainty


     INPUTS
    ========
     x         - ND array
     sigma   - uncertainty
     data      - data that that centres the l2 norm
    """
    def __init__(self, sigma, data, Phi):

        if np.any(sigma <= 0):
            raise Exception("'sigma' must be positive")
        self.sigma = sigma
        self.data = data
        self.beta = 1.
        if(np.any(Phi is None)):
            self.Phi = operators.identity
        else:
            self.Phi = Phi


    def grad(self, x):
        return self.Phi.adj_op((self.Phi.dir_op(x) - self.data))/self.sigma**2

    def fun(self, x):
        return np.sum(np.abs(self.data  - self.Phi.dir_op(x))**2.)/(2 * self.sigma**2)

