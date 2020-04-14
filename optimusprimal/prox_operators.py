import optimusprimal.linear_operators as linear_operators
import numpy as np

class l2_ball:
    """
    This class computes the proximity operator of the l2 ball

                        f(x) = (||Phi x - y|| < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of l2-ball
     data      - data that that centres the l2-ball
     Phi       - Measurement/Weighting operator
    """
    
    
    def __init__(self, epsilon, data, Phi=None):

        if np.any(epsilon <= 0):
            raise Exception("'epsilon' must be positive")
        self.epsilon = epsilon;
        self.data = data
        self.beta = 1.
        if(Phi is None):
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        xx = np.sqrt(np.sum( np.square(np.abs(x - self.data))))
        if (xx < self.epsilon):
            p = x
        else:
            p = (x - self.data) * self.epsilon / xx + self.data
        
        return p
        
    def fun(self, x):
        return 0;
    def dir_op(self, x):
        return self.Phi.dir_op(x)
    def adj_op(self, x):
        return self.Phi.adj_op(x)

class l1_norm:
    
    """
    This class computes the proximity operator of the l2 ball

                        f(x) = ||Psi x||_1 * gamma

    When the input 'x' is an array. gamma is a regularization term. Psi is a sparsity operator.


     INPUTS
    ========
     x         - ND array
     gamma     - regularization parameter
     Phi       - Measurement/Weighting operator
    """
    
    
    def __init__(self, gamma, Psi = None):

        if np.any( gamma <= 0 ):
            raise Exception("'gamma' must be positive")

        self.gamma = gamma
        self.beta = 1.

        if(Psi is None):
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    
    def prox(self, x, tau):
        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(complex(0, 1) * np.angle(x));
        
        
        
    def fun(self, x):
        return np.abs(self.gamma * x).sum();
    
    def dir_op(self, x):
        return self.Psi.dir_op(x)
    def adj_op(self, x):
        return self.Psi.adj_op(x)
        

class l2_square_norm:
    
    """
    This class computes the proximity operator of the l2 squared

                        f(x) = 0.5/sigma^2 * ||Psi x||_2^2

    When the input 'x' is an array. 0.5/sigma^2 is a regularization term. Psi is an operator.


     INPUTS
    ========
     x         - ND array
     sigma     - regularization parameter
     Phi       - Measurement/Weighting operator
    """
    
    
    def __init__(self, sigma, Psi = None):

        if np.any( sigma <= 0 ):
            raise Exception("'gamma' must be positive")

        self.sigma = sigma
        self.beta = 1.

        if(Psi is None):
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    
    def prox(self, x, tau):
        return x/(tau/self.sigma**2 + 1.);
        
        
        
    def fun(self, x):
        return np.sum(np.abs(x)**2/(2. * self.sigma**2));
    
    def dir_op(self, x):
        return self.Psi.dir_op(x)
    def adj_op(self, x):
        return self.Psi.adj_op(x)


class positive_prox:
    """
    This class computes the proximity operator of the indicator function for positivity

                        f(x) = (Re{x} >= 0) ? 0. : infty
    it returns the projection.

    """
    
    def __init__(self):
        self.beta = 1.


    
    def prox(self, x, tau):
        return np.maximum(0, np.real(x))
        
    def fun(self, x):
        return 0.;
    
    def dir_op(self, x):
        return x
    def adj_op(self, x):
        return x
        
class real_prox:
    """
    This class computes the proximity operator of the indicator function for reality

                        f(x) = (Re{x} == x) ? 0. : infty
    it returns the projection.

    """
    
    def __init__(self):
        self.beta = 1.

    def prox(self, x, tau):
        return np.real(x)
        
    def fun(self, x):
        return 0.;
    
    def dir_op(self, x):
        return x
    def adj_op(self, x):
        return x
        
    
