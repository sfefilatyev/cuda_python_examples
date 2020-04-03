"""
Conway game of life that simulates cellular automata and consists of the following rules:

* Any live cell with fewer than two live neighbors dies
* Any live cell with two or three neighbors lives
* Any live cell with more than three neighbors dies
* Any dead cell with exactly three neighbors comes to live
"""

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt
