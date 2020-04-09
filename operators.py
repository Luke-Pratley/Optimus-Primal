import numpy as np
class identity:
    """
    Identity operator

    """
    
    def dir_op(self, x):
        return x
    def adj_op(self, x):
        return x

class diag_matrix_operator:
    """
    Applies diagonal matrix operator W * x

     INPUTS
    ========
    W - array of weights
    """
    
    
    def __init__(self, W):
        self.W = W
    
    def dir_op(self, x):
        return self.W * x
    def adj_op(self, x):
        return np.conj(self.W) * x

class matrix_operator:
    """
    Applies matrix operator A * x

     INPUTS
    ========
    A - numpy matrix
    """
    
    
    def __init__(self, A):
        self.A = A
        self.A_H = np.conj(A.T)

    def dir_op(self, x):
        return self.A @ x
    def adj_op(self, x):
        return self.A_H @ x
