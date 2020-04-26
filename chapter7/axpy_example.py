import pycuda.autoinit
from pycuda import gpuarray
import numpy as np

from skcuda import cublas

a = np.float32(10)
x = np.float32([1, 2, 3])

y = np.float32([-.345, 8.15, -15.867])

x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

cublas_context_h = cublas.cublasCreate()

cublas.cublasSaxpy(cublas_context_h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)

cublas.cublasDestroy(cublas_context_h)

print("This is close to the NumPy approximation: {}".format(np.allclose(a * x + y, y_gpu.get())))
