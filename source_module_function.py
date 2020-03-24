import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule("""
__global__ void scalar_multipy_kernel(float *outvec, float scalar, float* vec)
{
    int i = threadIdx.x;
    outvec[i] = scalar * vec[i];   
}
""")
