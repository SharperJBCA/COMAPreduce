import numpy as np
cimport numpy as np
from cpython cimport array
import array

def mean_val(double[:] tod, long[:] mask):

    cdef int i
    cdef int nsamples = tod.size
    cdef top = 0
    cdef bottom = 0
    for i in range(nsamples):
        if mask[i] == 1:
            top += tod[i]
            bottom += 1

    return top, bottom

def running_mean(double[:] tod, long[:] mask, long step):

    cdef int i
    cdef int nsamples = tod.size
    
    cdef double top, bottom
    cdef double itop, ibottom
    for i in range(nsamples):
        if i < nsamples-step-1:
            itop, ibottom = mean_val(tod[i:i+step], mask[i:i+step])
        if ibottom > step/2:
            break

    for i in range(nsamples):
        if i < nsamples-step-1:
            top, bottom = mean_val(tod[i:i+step], mask[i:i+step])
        if bottom == 0:
            tod[i] -= itop/ibottom
        else:
            tod[i] -= top/bottom
                
                
