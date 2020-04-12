"""
Illustrate use of streams for syncronization.
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time

num_arrays = 200
array_len = 1024 ** 2;

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


data = []
data_gpu = []
gpu_out = []
streams = []
start_events = []
end_events = []

for _ in range(num_arrays):
    streams.append(drv.Stream())
    start_events.append(drv.Event())
    end_events.append(drv.Event())

# Generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

print("Starting GPU operations")

t_start = time()

# Copy arrays to GPU.
for k in range(num_arrays):
    data_gpu.append(gpuarray.to_gpu_async(data[k], stream=streams[k]))

# Process arrays.
for k in range(num_arrays):
    start_events[k].record(streams[k])
    mult_ker(data_gpu[k], np.int32(array_len), block=(32, 1, 1), grid=(1, 1, 1), stream=streams[k])

# could be in the previous loop, but that's what the book suggests.
for k in range(num_arrays):
    end_events[k].record(streams[k])

# Copy arrays from GPU.
for k in range(num_arrays):
    gpu_out.append(data_gpu[k].get_async(stream=streams[k]))

t_end = time()

print("Finishing GPU operations")

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

kernel_times = []
for k in range(num_arrays):
    kernel_times.append(start_events[k].time_till(end_events[k]))

print('Total time: {}'.format(t_end - t_start))
print("Mean kernel duration (milliseconds): {}".format(np.mean(kernel_times)))
print("Mean kernel standard deviation (milliseconds): {}".format(np.std(kernel_times)))
