"""
WORK EFFICIENT PARLLEL PREFIX ALGORITHM IMPLEMENTATION.

We have a binary operator @, that is a function that acts on two input values and gives one output 
value (such as +,*, V, ^) and a collection of elements X0, X1, X2, ,X3, ..., Xn-1. And from these we need to
compute efficiently X0 @ X1 @ X2 @ X3 @ ... @ Xn-1.
Naive parallel prefix algorithm is the origina version of this algorithm. It is naive b/c it makes an
assumption that given n input elements, n is dyadic ( n=2^k for some positive integer, k) and we can run
the algorithm in parallel over n processors (or n threads).
"""
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time

up_ker = SourceModule(
"""
__global__ void up_ker(double * x, double * x_old, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int _2k = 1 << k;
    int _2k1 = 1 << (k+1);

    int j = tid * _2k1;

    x[j + _2k1 - 1] = x_old[j + _2k - 1] + x_old[j + _2k1 - 1];
}
"""
)

up_gpu = up_ker.get_function("up_ker")

def up_sweep(x):
    x = np.float64(x)
    x_gpu = gpuarray.to_gpu(np.float64(x))
    x_old_gpu = x_gpu.copy()

    for k in range(int(np.log2(x.size))):
        num_threads = int(np.ceil(x.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))
        if grid_size > 1:
            block_size = 32
        else:
            blocK_size = num_threads
        up_gpu(x_gpu, x_old_gpu, np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        x_old_gpu[:] = x_gpu[:]

    x_out = x_gpu.get()
    return x_out

down_ker = SourceModule(
"""
__global__ void down_ker(double * y, double * y_old, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int _2k = 1 << k;
    int _2k1 = 1 << (k + 1);

    int j = tid * _2k1;

    y[j + _2k - 1] = y_old[j + _2k1 - 1];
    y[j + _2k1 - 1] = y_old[j + _2k1 - 1] + y_old[j + _2k -1 ];
}
"""
)

down_gpu = down_ker.get_function("down_ker")

def down_sweep(y):
    y = np.float64(y)
    y[-1] = 0
    y_gpu = gpuarray.to_gpu(y)
    y_old_gpu = y_gpu.copy()

    for k in reversed(range(int(np.log2(y.size)))):
        num_threads = int(np.ceil(y.size / 2**(k+1)))
        grid_size = int(np.ceil(num_threads / 32))
        if grid_size > 1:
            block_size = 32
        else:
            block_size = num_threads
        down_gpu(y_gpu, y_old_gpu, np.int32(k), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        y_old_gpu[:] = y_gpu[:]
    y_out = y_gpu.get()
    return y_out

def efficient_prefix(x):
    return down_sweep(up_sweep(x))


if __name__ == '__main__':
    testvec = np.random.randn(1024).astype(np.float64)

    outvec = efficient_prefix(testvec)

    total_sum = sum(testvec)
    total_sum_gpu = outvec[-1]

    print("Does our kernel work correctly? : {}".format(
        np.allclose(total_sum_gpu, total_sum)))
