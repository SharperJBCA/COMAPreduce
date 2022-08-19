import numpy as np
import Destriper
import COMAPData
import sys
from astropy.wcs import WCS

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == "__main__":

    nypix = 480
    nxpix = 480
    cdelt = [-0.016666,0.016666]
    crpix = [ 240,240]
    crval = [ 25.435, 0.0]
    ctype = ['RA---CAR', 'DEC--CAR']
    w = WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.cdelt = cdelt
    w.wcs.crpix = crpix
    w.wcs.ctype = ctype
    nxpix = nxpix
    nypix = nypix

    pixel_edges = np.arange(nxpix*nypix)

    filelist = np.loadtxt(sys.argv[1],dtype=str)
    
    step = filelist.size//size
    lo = step*rank
    hi = step*(rank+1)
    if hi > filelist.size:
        hi = filelist.size
    filelist = filelist[lo:hi]

    offset_length = 50
    feeds = [1,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19]
    tod, weights, pointing = COMAPData.read_comap_data(filelist,w,
                                                       offset_length=offset_length,
                                                       feeds=feeds)
    run_destriper(pointing,tod,weights,offset_length,pixel_edges)
