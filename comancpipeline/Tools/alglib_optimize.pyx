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

cdef extern from "src/mincg_d_1.cpp" nogil:
    vector[double] run_fit(vector[double] &ix, 
                           vector[double] &iy,
                           vector[double] &iz,
                           vector[double] &iw, 
                           vector[double] &ic,
                           vector[double] &ie,
                           int& maxits,
                           double& epsx,
                           double& diffstep)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] vec2view(vector[double]& v):
    cdef int i
    cdef double[::1] out = np.empty(v.size())
    for i in range(v.size()):
        out[i] = v[i];
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:] vec2view_2d(vector[vector[double]]& v):
    cdef int i, j
    cdef double[:,:] out = np.empty((v.size(),v[0].size()))
    cdef icnt = v.size()
    cdef jcnt = v[0].size()
    for i in range(icnt):
        for j in range(jcnt):
            out[i][j] = v[i][j];
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[double] view2vec(double[:]& v):
    cdef int i
    cdef vector[double] out
    for i in range(v.size):
        out.push_back(v[i]);
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[double] fit_wrap(vector[double]& x,
                             vector[double]& y, 
                             vector[double]& z, 
                             vector[double]& w, 
                             vector[double]& c,
                             vector[double]& e,
                             int& maxits,
                             double& epsx,
                             double& diffstep) nogil:
    return run_fit(x,y,z,w,c,e,maxits, epsx, diffstep)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double median_val(vector[double]& z, int& start, int& stop) nogil:
    cdef double top, bot, mval
    top = 0
    bot = 0
    mval = 0
    for i in range(start,stop):
        top = top + z[i]
        bot = bot + 1.0
    
    mval = top/bot
    return mval

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void remove_trend(vector[double]& z, int& step) nogil:
    # remove a linear trend from the data
    cdef int i
    cdef int nsamples = z.size()
    cdef int nsteps = nsamples/step
    
    cdef int counter = 0
    cdef int istep   = 0
    cdef int start, stop

    cdef double medval

    start = 0
    stop = step
    medval = median_val(z, start, stop)
    for i in range(nsamples):
        istep = i/step
        start = step*istep
        if start > nsamples:
            break

        stop  = step*(istep+1)
        if stop > nsamples:
            stop = nsamples
        if counter == 0: # if the counter is starting, calc the median
            medval = median_val(z, start, stop)
        
        z[i] = z[i] - medval
        #printf("%d ", i)
        #printf("%d ", counter)
        #printf("%d ", start)
        #printf("%d\n", stop)
        if counter < (step-1):
            counter = counter + 1
        else:
            #istep = istep + 1
            counter = 0



@cython.boundscheck(False)
@cython.wraparound(False)
def mean_filt(double[:] vz, int _step):

    cdef int step = _step

    cdef vector[double] z = view2vec(vz)
    remove_trend(z, step)
    return vec2view(z)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(double[:] vx, double[:] vy, double[:,:] vz, double[:] vw, double[:] vc,
         int _step, int _maxits, double _epsx, double _diffstep):

    cdef int nsize = vz.shape[0]
    cdef int i
    cdef int step = _step
    cdef int maxits = _maxits
    cdef double epsx = _epsx
    cdef double diffstep = _diffstep

    cdef int tid
    cdef vector[vector[double]] params
    params.resize(vz.shape[0])
    for i in range(nsize):
        params[i].resize(5)


    cdef vector[vector[double]] z 
    z.resize(vz.shape[0])

    cdef vector[double] x = view2vec(vx)
    cdef vector[double] y = view2vec(vy)

    for i in range(nsize):
        z[i] = view2vec(vz[i])
    cdef vector[double] w = view2vec(vw)
    cdef vector[double] c = view2vec(vc)

    cdef vector[vector[double]] e
    e.resize(vz.shape[0])
    for i in range(nsize):
        e[i].resize(5)

    for i in prange(nsize,nogil=True,schedule='guided'):
        remove_trend(z[i], step)
        #print(c[0],vc[0],x[0])
        params[i] = fit_wrap(x,y,z[i],w,c,e[i], maxits, epsx, diffstep)
        #print(z[i][0],params[i][0],c[0],vc[0])
        #printf("%d ", i)
        #tid = threadid()
        #printf("%f ", z[i][0])
        #printf("%f ", params[i][0])
        #printf("%d\n", tid)
    
    return vec2view_2d(params), vec2view_2d(e)
