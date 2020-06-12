from __future__ import division
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from cuda_driver import *

def mandelbrot(breadth, low, high, max_iters, upper_bound):
    cuInit(0)
    cnt = c_int(0)
    cuDeviceGetCount(byref(cnt))
    if cnt.value == 0:
        raise Exception('No GPU device found')

    cuDevice = c_int(0)
    cuDeviceGet(byref(cuDevice), 0)
    cuContext = c_void_p()
    cuCtxCreate(byref(cuContext), 0, cuDevice)

    cuModule = c_void_p()
    cuModuleLoad(byref(cuModule), c_char_p('./mandelbrot.ptx'))

    
