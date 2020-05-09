# This scripts implements calculation of definite integrations using Monte Carlo sampling method.

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
