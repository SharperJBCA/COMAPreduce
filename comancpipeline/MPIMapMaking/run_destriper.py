import numpy as np
import Destriper
import COMAPData
import sys
from astropy.wcs import WCS
from matplotlib import pyplot
from astropy.io import fits
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from comancpipeline.Tools import Coordinates,ParserClass

def write_map(prefix,maps,map_info,output_dir,iband,postfix=''):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #print(output_dir)
    #stop
    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']

    for k,v in maps.items():
        print(k)
        hdulist = []
        hdu = fits.PrimaryHDU(np.reshape(v['map'],(nypix,nxpix)),
                              header=wcs.to_header())
        hdulist += [hdu]

        if 'naive' in v:
            naive = fits.ImageHDU(np.reshape(v['naive'],(nypix,nxpix)),
                                  name='Naive',header=wcs.to_header())
            hdulist += [naive]
        if 'weight' in v:
            cov = fits.ImageHDU(np.reshape(np.sqrt(1./v['weight']),(nypix,nxpix)),
                                name='Noise',header=wcs.to_header())
            hdulist += [cov]

        if 'map2' in v:
            std = fits.ImageHDU(np.reshape(np.sqrt(v['map2']-v['map']**2),(nypix,nxpix)),
                                name='NoiseSTD',
                                header=wcs.to_header())
            hdulist += [std]
        hdul = fits.HDUList(hdulist)
        fname = '{}/{}_{}_Band{:02d}.fits'.format(output_dir,k,prefix,iband)
        hdul.writeto(fname,overwrite=True)


def main(filelistname,
         offset_length = 50,
         tod_name='filtered_tod',
         feed_weights=None,
         prefix = 'fg9',
         output_dir = 'maps/fg9/',
         obsid_cuts = [],
         feeds = [1,2,3,5,6,9,11,12,13,14,15,16,17,18,19],
         nxpix=480,
         nypix=480,
         crval = [Coordinates.sex2deg('05:32:00.3',hours=True),
                  Coordinates.sex2deg('+12:30:28.0')], # fg9
         crpix=[ 240,240],
         ctype = ['RA---CAR', 'DEC--CAR'],
         cdelt=[-0.016666,0.016666]):

    filelist = np.loadtxt(filelistname,dtype=str,ndmin=1)

    if isinstance(crval[0],str):
        crval = [Coordinates.sex2deg(c,hours=f) for c,f in zip(crval,[True,False])]

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
    
    step = filelist.size//size
    #lo = step*rank
    #hi = step*(rank+1)
    #if hi > filelist.size:
    #    hi = filelist.size

    select = np.mod(np.arange(filelist.size),size)
    select = np.sort(np.where((select == rank))[0]).astype(int)
    lo = select[0]
    hi = select[0]+1
    
    print(rank,filelist.size)
    filelist = filelist[select]
    print(lo,hi,filelist.size)

    for iband in range(0,4):
        tod, weights, pointing, az, el ,feedid, obsids = COMAPData.read_comap_data(filelist,map_info,
                                                                                   tod_name=tod_name,
                                                                                   feed_weights=feed_weights,
                                                                                   offset_length=offset_length,
                                                                                   iband=iband,
                                                                                   feeds=feeds)

        maps = Destriper.run_destriper(pointing,
                                       tod,
                                       weights,
                                       offset_length,
                                       pixel_edges,
                                       az,
                                       el,
                                       feedid,
                                       obsids,
                                       obsid_cuts,
                                       np.array(feeds).astype(int),
                                       chi2_cutoff=30)

        #maps = {}
        #map_info = {}
        if rank == 0:
            write_map(prefix,maps,map_info,output_dir,iband,postfix='')
        comm.Barrier()

if __name__ == "__main__":
    
    p = ParserClass.Parser(sys.argv[1],delims=['='])
    params = p['Inputs']
    main(params['filelistname'],
         obsid_cuts = [[8590,12967],[12985,15336],[15441,20355]],
         tod_name=params['tod_name'],
         offset_length=int(params['offset_length']),
         prefix=params['prefix'],
         output_dir=params['output_dir'],
         feeds=params['feeds'],
         feed_weights=params['feed_weights'],
         nxpix=int(params['nxpix']),
         nypix=int(params['nypix']),
         crval=params['crval'],
         crpix=params['crpix'],
         ctype=params['ctype'],
         cdelt=params['cdelt'])
