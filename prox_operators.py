import operators
import numpy as np

class L2_Ball:
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
            self.Phi = operators.identity()
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
        return self.Phi.forward(x)
    def adj_op(self, x):
        return self.Phi.adjoint(x)

class l1_norm_prox :
    
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
    
    
    def __init__(self, gamma, Psi):

        if np.any( gamma <= 0 ):
            raise Exception("'gamma' must be positive")

        self.gamma = gamma
        self.beta = 1.

        if(Phi is None):
            self.Phi = operators.identity()
        else:
            self.Phi = Phi

    
    def prox(self, x, tau):
        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(complex(0, 1) * np.angle(x));
        
        
        
    def fun(self, x):
        return np.abs(self.gamma * x).sum();
    
    def dir_op(self, x):
        return self.Psi.dir_op(x)
    def adj_op(self, x):
        return self.Psi.adj_op(x)
        
        
 class positive_prox:
    
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
        
        
    
