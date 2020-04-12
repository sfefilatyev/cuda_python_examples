"""
This script adds a simple example of manual context creation.
Copies a small array to the GPU and copies it back to the host, prints it, exists.
"""

import numpy as np
from pycuda import gpuarray
import pycuda.driver as drv

# Cuda initialization.
drv.init()

# Chosing GPU for the new context.
dev = drv.Device(0)

# Create context on device 0.
ctx = dev.make_context()

# New context will automatically become default.
x = gpuarray.to_gpu(np.float32([1, 2, 3]))
print(x.get())

# Destroy context.
ctx.pop()
