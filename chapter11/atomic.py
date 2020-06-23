from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv

AtomicCode='''
__global__ void atomic_ker(int* add_out, int* max_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicExch(add_out, 0);
    __syncthreads();
    atomicAdd(add_out, 1);
    atomicMax(max_out, tid);
}
'''

atomic_mod = SourceModule(AtomicCode)
atomic_ker = atomic_mod.get_function('atomic_ker')

add_out = gpuarray.empty((1,), dtype=np.int32)
max_out = gpuarray.empty((1,), dtype=np.int32)

atomic_ker(add_out, max_out, grid=(1, 1, 1), block=(100, 1, 1))

print("Atomic operations test:")
print("add_out: {}".format(add_out.get()[0]))
print("max_out: {}".format(max_out.get()[0]))
