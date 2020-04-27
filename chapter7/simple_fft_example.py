import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft

x = np.asarray(np.random.rand(1000), dtype=np.float32)
x_gpu = gpuarray.to_gpu(x)
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)
