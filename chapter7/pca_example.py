# This example demonstrates simple PCA-computation using CUDA-Solver library.

import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import linalg

vals = [np.float32([10, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.float32([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])]

for i in range(3000):
    vals.append(vals[0] + 0.001 * np.random.randn(10))
    vals.append(vals[1] + 0.001 * np.random.randn(10))
    vals.append(0.001 * np.random.randn(10))

vals = np.float32(vals)
vals = vals - np.mean(vals, axis=0)
v_gpu = gpuarray.to_gpu(vals.T.copy())

U_d, s_d, V_d = linalg.svd(v_gpu, lib='cusolver')

u = U_d.get()
s = s_d.get()
v = V_d.get()

print(s**2)
print(u[:,0])
print(u[:,1])
