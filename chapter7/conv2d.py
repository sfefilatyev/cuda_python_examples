# This example implements simple 2D convolution routine using CUDA FFT backend.
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import fft
from skcuda improt linalg
from matplotlib import pyplot as plt
