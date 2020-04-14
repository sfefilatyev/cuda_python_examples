"""
Illustrate use of multiple kernels without streams.
"""
import pycuda
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from time import time
import threading

num_arrays = 10
array_len = 1024 ** 2;

kernel_code = \
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

class KernelLauncherThread(threading.Thread):
    def __init__(self, input_array):
        threading.Thread.__init__(self)
        self.input_array = input_array
        self.output_array = None

    def run(self):
        self.dev = drv.Device(0)
        self.context = self.dev.make_context()
        self.ker = SourceModule(kernel_code)
        self.mult_ker = self.ker.get_function('mult_ker')

        self.array_gpu = gpuarray.to_gpu(self.input_array)
        self.mult_ker(self.array_gpu, np.int32(array_len), block=(64,1,1), grid=(1,1,1))
        self.output_array = self.array_gpu.get()
        self.context.pop()

    def join(self):
        threading.Thread.join(self)
        return self.output_array


data = []
gpu_out = []
threads = []

drv.init()

# Generate random arrays.
for _ in range(num_arrays):
    data.append(np.random.randn(array_len).astype('float32'))

for k in range(num_arrays):
    threads.append(KernelLauncherThread(data[k]))

for k in range(num_arrays):
    threads[k].start()

for k in range(num_arrays):
    gpu_out.append(threads[k].join())

for k in range(num_arrays):
    assert (np.allclose(gpu_out[k], data[k]))

