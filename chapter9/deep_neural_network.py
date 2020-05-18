from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pucuda.compiler import SourceModule
from pucuda.elementwise import ElementwiseKernel
import numpy as np
from Queue import Queue
import csv
import time

dense_eval_code='''
__global__ void dense_eval(int num_outputs, int num_inputs, int relu, int sigmoid, float* w, float* b, float* x, float* y, int batch_size, int w_t, int_bt, float delta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_output)
    {
        for (int k = 0; k < batch_size; k++)
        {
            double temp = 0.0f;
            for (int j = 0; j < num_inputs; j++)
            {
                
            }
        }
    }
}
'''
