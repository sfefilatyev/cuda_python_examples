# This simplest example shows how to load a 'printf' function from a libc.so shared library usign python.

import ctypes
libc = ctypes.CDLL('/lib/x86_64-linux-gnu/libc-2.27.so')
libc.printf("Hello from ctypes!\n")
libc.printf("Pi is approximately %f.\n", ctypes.c_double(3.14))
