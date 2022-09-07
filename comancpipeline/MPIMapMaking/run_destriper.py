import numpy as np
import Destriper
import COMAPData
import sys
from astropy.wcs import WCS
from matplotlib import pyplot
from astropy.io import fits

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from comancpipeline.Tools import Coordinates

def write_map(maps,map_info,postfix=''):

    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']
    hdu = fits.PrimaryHDU(np.reshape(maps['map'],(nxpix,nypix)),
                                     header=wcs.to_header())
    #cov = fits.ImageHDU(variance,name='Covariance',header=wcs.to_header())
    #hits = fits.ImageHDU(hits,name='Hits',header=wcs.to_header())
    naive = fits.ImageHDU(np.reshape(maps['naive'],(nxpix,nypix)),
                          name='Naive',header=wcs.to_header())

    hdul = fits.HDUList([hdu,naive])
    fname = 'co2.fits'
    hdul.writeto(fname,overwrite=True)


if __name__ == "__main__":

    nypix = 480
    nxpix = 480
    cdelt = [-0.016666,0.016666]
    crpix = [ 240,240]
    crval = [Coordinates.sex2deg('00:42:44.330',hours=True),
             Coordinates.sex2deg('+41:16:07.50')]
    #[ 25.435, 0.0]
    ctype = ['RA---CAR', 'DEC--CAR']
    w = WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.cdelt = cdelt
    w.wcs.crpix = crpix
    w.wcs.ctype = ctype
    nxpix = nxpix
    nypix = nypix

    map_info = {'wcs':w,
                'nxpix':nxpix,
                'nypix':nypix}

    pixel_edges = np.arange(nxpix*nypix)

    filelist = np.loadtxt(sys.argv[1],dtype=str,ndmin=1)
    
    step = filelist.size//size
    lo = step*rank
    hi = step*(rank+1)
    if hi > filelist.size:
        hi = filelist.size
    print(lo,hi)
    filelist = filelist[lo:hi]

    offset_length = 50
    feeds = [1,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19]
    tod, weights, pointing = COMAPData.read_comap_data(filelist,map_info,
                                                       offset_length=offset_length,
                                                       feeds=feeds)
    maps = Destriper.run_destriper(pointing,tod,weights,
                                   offset_length,pixel_edges)

    if rank == 0:
        write_map(maps,map_info,postfix='')
        pyplot.imshow(np.reshape(maps['map'],(nxpix,nypix)))
        pyplot.savefig('test.png')
