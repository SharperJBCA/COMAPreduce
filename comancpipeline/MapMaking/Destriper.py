"""
Destriper.py -- An MPI ready implementation of the Destriping algorithm.

Includes a test script + some methods simulating noise and signal

run Destriper.test() to run example script.

Requires a wrapper that will creating the pointing, weights and tod vectors
that are needed to pass to the Destriper.

This implementation does not care about the coordinate system

Refs:
Sutton et al. 2011 

"""
import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import gaussian_filter
from comancpipeline.Tools import binFuncs
import healpy as hp
import sys
import shutil
import psutil

from comancpipeline.MapMaking.mpi_functions import sum_map_all_inplace, mpi_sum

from comancpipeline.MapMaking import CBASSExperiment as Experiment
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def cgm(pointing, pixel_edges, tod, weights, obsids, A,b,x0 = None,niter=100,threshold=1e-3,verbose=False, offset_length=50):
    """
    Biconjugate CGM implementation from Numerical Recipes 2nd, pg 83-85
    
    arguments:
    b - Array
    Ax - Function that applies the matrix A
    
    kwargs:
    
    Notes:
    1) Ax can be a class with a __call__ function defined to act like a function
    2) Weights should be average weight for an offset, e.g. w_b = sum(w)/offset_length
    3) Need to add a preconditionor matrix step
    """
    
    
    if isinstance(x0,type(None)):
        x0 = np.zeros(b.size,dtype=np.float32)
    
    r  = b - A.matvec(x0)
    rb = b - A.matvec(x0)
    p  = r*1.
    pb = rb*1.

    thresh0 = mpi_sum(r*rb) 
    for i in range(niter):
        
        q = A.matvec(pb)


        rTrb = mpi_sum(r*rb) 
        alpha= rTrb/mpi_sum(pb*q)

        x0 += alpha*pb
        
        r = r - alpha*A.matvec(p)
        rb= rb- alpha*A.matvec(pb)
        
        beta = mpi_sum(r*rb)/rTrb
        
        p = r + beta*p
        pb= rb+ beta*pb
        
        delta = mpi_sum(r*rb)/thresh0
        
        if verbose:
            print(delta)
        if rank ==0:
            print(delta, threshold,flush=True)
        if delta < threshold:
            break
        

    if rank == 0:
        if (i == (niter-1)):
            print('Convergence not achieved in {} steps'.format(niter),flush=True)

        print('Final covergence: {} in {:d} steps'.format(delta,i),flush=True)

    return x0

def run(pointing,tod,weights,offset_length,pixel_edges, obsids, special_weight=None):

    print(psutil.Process().memory_info().rss/(1024*1024))

    i_op_Ax = Experiment.op_Ax(pointing,weights,offset_length,pixel_edges,
                    special_weight=special_weight)

    b = i_op_Ax(tod,extend=False)

    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax, dtype=np.float32)

    print('A memory', sys.getsizeof(A) / (1024 * 1024)) 
    print('Ax memory', sys.getsizeof(i_op_Ax) / (1024 * 1024)) 
    print('tod memory', sys.getsizeof(tod) / (1024 * 1024)) 
    print('weights memory', sys.getsizeof(weights) / (1024 * 1024)) 
    print('pointing memory', sys.getsizeof(pointing) / (1024 * 1024)) 
    print('obsid memory', sys.getsizeof(obsids) / (1024 * 1024)) 
    print('b memory', sys.getsizeof(b) / (1024 * 1024)) 

    for k, v in i_op_Ax.__dict__.items():
        print(f'i_op_Ax.{k} memory', sys.getsizeof(v) / (1024 * 1024)) 

    print(psutil.Process().memory_info().rss/(1024*1024))
    if rank == 0:
        print('Starting CG',flush=True)
    if True:
        x = cgm(pointing, pixel_edges, tod, weights, obsids, A,b, offset_length=offset_length,threshold=1e-3)
    else:
        x= np.zeros(b.size)
    if rank == 0:
        print('Done',flush=True)
    return x, i_op_Ax

def destriper_iteration(_pointing,
                        _tod,
                        _weights,
                        offset_length,
                        pixel_edges,
                        obsids,
                        special_weight=None):
    if isinstance(special_weight,type(None)):
        special_weight = np.ones(_tod.size)

    result,i_op_Ax = run(_pointing,_tod,_weights,offset_length,pixel_edges,
                 obsids,
                 special_weight=special_weight)
    
    maps = Experiment.sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax)
    return maps, result

def run_destriper(_pointing,
                  _tod,
                  _weights,
                  offset_length,
                  pixel_edges,
                  obsids,
                  chi2_cutoff=100,special_weight=None,healpix=False):

    _maps,result = destriper_iteration(_pointing,
                                      _tod,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      obsids,
                                      special_weight=special_weight)

    maps = {'All':_maps}
    offsets = {'All':np.repeat(result,offset_length)}

    return maps, offsets 