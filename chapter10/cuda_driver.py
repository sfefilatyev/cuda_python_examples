from ctypes import *
import sys
if 'linux' in sys.platform:
    cuda = CDLL('libcuda.so')
elif 'win' in sys.platform:
    cuda = CDLL('nvcuda.dll')

cuInit = cuda.cuInit
cuInit.argtypes = [c_uint]
cuInit.restypes = int

cuDeviceGetCount = cuda.cuDeviceGetCount
cuDeviceGetCount.argtypes = [POINTER(c_int)]
cuDeviceGetCount.restype = int

cuDeviceGet = cuda.cuDeviceGet
cuDeviceGet.argtypes = [POINTER(c_int), c_int]
cuDeviceGet.restype = int

cuCtxCreate = cuda.cuCtxCreate
cuCtxCreate.argtypes = [c_void_p, c_uint, c_int]
cuCtxCreate.restype = int

cuModuleLoad = cuda.cuModuleLoad
cuModuleLoad.argtypes = [c_void_p, c_char_p]
cuModuleLoad.restype = int

cuCtxSyncronize = cuda.cuCtxSyncronize
cuCtxSyncronize.argtypes = []
cuCtxSyncronize.restype = int

cuModuleGetFunction = cuda.cuModuleGetFunction
cuModuleGetFunction.argtypes = [c_void_p, c_void_p, c_char_p]
cuModuleGetFunction.restype = int
