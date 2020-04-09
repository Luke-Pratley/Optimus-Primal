import numpy as np
import pywt
class identity:
    """
    Identity operator

    """
    
    def dir_op(self, x):
        return x
    def adj_op(self, x):
        return x
class function_wrapper(dir_op, adj_op):
    """
    Given direct and adjoint functions return linear operator

     INPUTS
    ========
    dir_op  - forward operator 
    adj_op  - adjoint operator
    """
    def __init__(self, dir_op, adj_op):
        self.dir_op = dir_op
        self.adj_op = adj_op

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


class db_wavelets:
    
    def __init__(self, wav, levels, shape):

        if np.any( levels <= 0 ):
            raise Exception("'levels' must be positive")
        self.wav = wav
        self.levels = levels
        self.shape = shape
        self.coeff_slices = None
        self.coeff_shapes = None

        self.adj_op(self.dir_op(np.ones(shape)))
        
    def dir_op(self, x):
        if (self.wav == "dirac"):
            return np.ravel(x)
        coeffs = pywt.wavedecn(x, wavelet=self.wav, level=self.levels, mode ='periodic')
        arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(coeffs)
        return arr
    def adj_op(self, x):
        if (self.wav == "dirac"):
            return np.reshape(x,self.shape)
        coeffs_from_arr = pywt.unravel_coeffs(x, self.coeff_slices, self.coeff_shapes, output_format='wavedecn')
        return pywt.waverecn(coeffs_from_arr, wavelet=self.wav, mode ='periodic')

class dictionary:
    sizes = []
    wavelet_list = []
    
    def __init__(self, wav, levels, shape = None):

        if np.any( levels <= 0 ):
            raise Exception("'levels' must be positive")
        self.wavelet_list = []
        self.sizes = np.zeros(len(wav))
        for i in range(len(wav)):
            self.wavelet_list.append(db_wavelets(wav[i], levels, shape))

    def dir_op(self, x):
        out = self.wavelet_list[0].dir_op(x)
        self.sizes[0] = out.shape[0]
        for wav_i in range(1, len(self.wavelet_list)):
            buff = self.wavelet_list[wav_i].dir_op(x)
            self.sizes[wav_i] = buff.shape[0]
            out = np.concatenate((out, buff), axis=0)
        return out/ np.sqrt(len(self.wavelet_list))
    def adj_op(self, x):
        offset = 0
        out = 0
        for wav_i in range(len(self.wavelet_list)):
            size = self.sizes[wav_i]
            x_block = x[int(offset):int(offset + size)]
            buff = self.wavelet_list[wav_i].adj_op(x_block)
            out += buff / np.sqrt(len(self.wavelet_list))
            offset += size
        return out
