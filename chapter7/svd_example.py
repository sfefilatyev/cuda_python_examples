# This example demonstrates simple SVD-computation using CUDA-Solver library.

import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import linalg

a = np.random.rand(1000, 5000).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)

U_d, s_d, V_d = linalg.svd(a_gpu, lib='cusolver')

U = U_d.get()
s = s_d.get()
V = V_d.get()

S = np.zeros((1000, 5000))
S[:1000, :1000] = np.diag(s)

print('Can we reconstruct a from its SVD decomposition? : {}'.format(np.allclose(a, np.dot(U, np.dot(S, V), atol=1e-5))))
