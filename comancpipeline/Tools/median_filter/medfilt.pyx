# distutils: language = c++

cimport cython
from cython.view cimport array as cvarray
from cpython cimport array as carray

from libcpp.vector cimport vector

from cython.parallel import prange, parallel, threadid

import numpy as np
cimport numpy as np

cimport openmp

from libc.math cimport sqrt, acos, sin, cos, floor, exp, pow
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
        
cdef extern from "medianFilter.cpp" nogil:
    void filter( double* array, int n, int filterSize )
    void filter_mask( double* array, int* mask, int n, int filterSize )

@cython.boundscheck(False)
@cython.wraparound(False)
def medfilt(double[::1] data, int filterSize):
    
    cdef int n = data.size
    cdef double *data_temp = &data[0]

    filter(data_temp, n, filterSize)

    return data


@cython.boundscheck(False)
@cython.wraparound(False)
def medfilt_mask(double[::1] data, int filterSize, int[::1] mask):
    
    cdef int n = data.size
    filter_mask(&data[0], &mask[0], n, filterSize)
    
    return data
