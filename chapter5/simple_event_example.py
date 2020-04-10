"""
Illustrate use of a simple event.
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

array_len = 100 * 1024 ** 2;

ker = SourceModule(
"""
__global__ void mult_ker(float * array, int array_len)
{
    int thd = blockIdx.x * blockDim.x + threadIdx.x;
    int num_iters = array_len / blockDim.x;

    for(int j = 0; j < num_iters; j++)
    {
        int i = j * blockDim.x + thd;
        for (int k = 0; k < 50; k++)
        {
            array[i] *= 2.0;
            array[i] /= 2.0;
        }
    }
}
"""
)

mult_ker = ker.get_function('mult_ker')

# Generate random array.
data = np.random.randn(array_len).astype('float32')

# Copy array to GPU.
data_gpu = gpuarray.to_gpu(data)

print("Starting GPU operations")

start_event = drv.Event()
end_event = drv.Event()

t_start = time()

start_event.record()

# Process array.
mult_ker(data_gpu, np.int32(array_len), block=(64, 1, 1), grid=(1, 1, 1))

end_event.record()

gpu_out = data_gpu.get()

t_end = time()

print("Finishing GPU operations")

print('Total time: {}'.format(t_end - t_start))
