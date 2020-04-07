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
