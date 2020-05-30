# This simplest example shows how to load a 'printf' function from a libc.so shared library usign python.

import ctypes
libc = ctypes.CDLL('/lib/i386-linux-gnu/libc.so.6')
libc.printf("Hellow from ctypes!\n")
libc.printf("Pi is approximately %f.\n", ctypes.c_double(3.14))
