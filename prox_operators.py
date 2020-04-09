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
    
    
    def __init__(self, epsilon, data, Phi = None):

        if np.any(epsilon <= 0 ):
            raise Exception("'epsilon' must be positive")
        self.epsilon = epsilon;
        self.data = data
        self.beta = 1.
        if(Phi = None):
            self.Phi = identity()
        else:
            self.Phi = Phi

    
    def prox(self, x, gamma):
        
        xx = np.sqrt(np.sum( np.square(np.abs(x - self.data))))
        if (xx < self.epsilon):
            p  = x 
        else:
            p =  (x - self.data) * self.epsilon /xx  + self.data
        
        return p
        
    def fun(self, x):
        return 0;
    def dir_op(self, x):
        return self.Phi.forward(x)
    def adj_op(self, x):
        return self.Phi.adjoint(x)
