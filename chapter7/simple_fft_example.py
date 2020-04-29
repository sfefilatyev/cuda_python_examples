import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft

x = np.asarray(np.random.rand(1000), dtype=np.float32)
x_gpu = gpuarray.to_gpu(x)
x_hat = gpuarray.empty_like(x_gpu, dtype=np.complex64)

plan = fft.Plan(x_gpu.shape, np.float32, np.complex64)
inverse_plan = fft.Plan(x.shape, in_dtype=np.complex64, out_dtype=np.float32)

fft.fft(x_gpu, x_hat, plan)
fft.ifft(x_hat, x_gpu, inverse_plan, scale=True)

y = np.fft.fft(x)

print('cuFFT matches NumPy FFT: {}'.format(np.allclose(x_hat.get(), y, atol=1e-6)))
print('cuFFT inverse matches original: {}'.format(np.allclose(x_gpu.get(), x, atol=1e-6)))

