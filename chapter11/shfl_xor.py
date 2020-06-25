from __future__ import division
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray

ShflCode='''
__global__ void shfl_xor_ker(int* input, int* output)
{
    int temp = input[threadIdx.x];
    temp = __shfl_xor(temp, 1, blockDim.x);
    output[threadIdx.x] = temp;
}
'''

shfl_mod = SourceModule(ShflCode)
shfl_ker = shfl_mod.get_function('shfl_xor_ker')

dinput = gpuarray.to_gpu(np.int32(range(32)))
doutput = gpuarray.empty_like(dinput)

shfl_ker(dinput, doutput, grid=(1, 1, 1), block=(32, 1, 1))

print('Input array: {}'.format(dinput.get()))
print('Array after __shfl_xor: {}'.format(doutput.get()))
