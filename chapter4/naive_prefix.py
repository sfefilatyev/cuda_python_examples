"""
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

naive_ker = SourceModule(
"""
__global__ void naive_prefix(double * vec, double * out)
{
    __shared__ double sum_buf[1024];
    int tid = threadIdx.x;
    sum_buf[tid] = vec[tid];

    int iter = 1;
    for(int i = 0; i < 10; i++)
    {
        __syncthreads();
        if (tid >= iter)
        {
            sum_buf[tid] = sum_buf[tid] + sum_buf[tid - iter];
        }
        iter *= 2;
    }
    __syncthreads();
    out[tid] = sum_buf[tid];
    __syncthreads();
}
"""
)

naive_gpu = naive_ker.get_function("naive_prefix")

if __name__ == '__main__':
    testvec = np.random.randn(1024).astype(np.float64)
    testvec_gpu = gpuarray.to_gpu(testvec)

    outvec_gpu = gpuarray.empty_like(testvec_gpu)
    naive_gpu(testvec_gpu, outvec_gpu, block=(1024, 1, 1), grid=(1, 1, 1))

    total_sum = sum(testvec)
    total_sum_gpu = outvec_gpu[-1].get()

    print("Does our kernel work correctly? : {}".format(
        np.allclose(total_sum_gpu, total_sum)))
