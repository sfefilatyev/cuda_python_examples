# This example demonstrates simple PCA-computation using CUDA-Solver library.

import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import linalg
