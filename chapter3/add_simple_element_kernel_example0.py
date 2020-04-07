# Add simple element-wise kernl PyCUDA example.

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from time import time
from pycuda.elementwise import ElementwiseKernel

host_data = np.float32(np.random.random(50000000))
gpu_2x_ker = ElementwiseKernel(
    "float* in, float* out",
    "out[i] = 2 * in[i];",
    "gpu_2x_ker")

def speed_comparison():
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()

    print("Total time to compute on CPU: {}".format(t2 - t1))

    device_data = gpuarray.to_gpu(host_data)
    # Allocate memory for output.
    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time() 

    from_device = device_data_2x.get()

    print("Total time to compute on GPU: {}".format(t2 - t1))
    print("Is the host computation the same as the GPU computation?: {}".format(
        np.allclose(from_device, host_data_2x)))


if __name__ == '__main__':
    speed_comparison()
    speed_comparison()
    speed_comparison()
