from CBASSData import CBASSData, CBASSDataSim, read_comap_data, read_comap_data_iqu
import numpy as np
from matplotlib import pyplot 
from astropy.io import fits 
import glob 
from tqdm import tqdm
import Destriper
import os 
import healpy as hp 
from datetime import datetime 
from astropy.time import Time 

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def write_map_iqu_healpix(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        hdulist = []
        m = np.zeros((6,12*nside**2)) + hp.UNSEEN
        npix = 12*nside**2
        m[0,:] = v['map'][:npix]
        m[1,:] = v['map'][npix:2*npix]
        m[2,:] = v['map'][2*npix:]
        m[3,:] = v['weight'][:npix]
        m[4,:] = v['weight'][npix:2*npix]
        m[5,:] = v['weight'][2*npix:]
        fname = '{}/{}_{}.fits'.format(output_dir,k,prefix)
        hp.write_map(fname,m, overwrite=True,  partial=False)

def write_map_healpix(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        hdulist = []
        m = np.zeros((3,12*nside**2)) + hp.UNSEEN
        m[0,:] = v['map']
        m[1,:] = v['naive']
        m[2,:] = np.sqrt(1./v['weight'])
        fname = '{}/{}_{}.fits'.format(output_dir,k,prefix)
        hp.write_map(fname,m, overwrite=True,  partial=False)

if __name__ == "__main__":
    datadir = '/scratch/nas_cbassarc/cbass_data/Reductions/v34m3/v34m3_gsS_tod/mcal1/'
    filename = f'{datadir}/24-Mar-2014:12:40:39_reload_mcal1.fits'

    filelist = np.loadtxt('mcal1_files_sorted_cut.txt',dtype=str)
    filelist = filelist[:filelist.size//2]
    filelist = filelist[:50]
    # with open('mcal1_files.txt','w') as f:
    #     for filename in tqdm(filelist):
    #         hdu = fits.open(filename,memmap=False)
    #         if 'NM20S3M1' in hdu:
    #             f.write(filename+'\n')
    #         hdu.close()
    nfile = len(filelist)
    idx = np.sort(np.mod(np.arange(nfile),size))
    idx = np.where(idx == rank)[0] 
    lo = idx[0]
    hi = idx[-1]+1
    
    nside = 128 
    offset_length = 500  # 100 samples = 1 second 
    #tod, weights, pointing, obsid, cbass_data, special_weights = read_comap_data(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length)
    tod, weights, pointing, obsid,special_weights, special_weights_rot, cbass_data = read_comap_data_iqu(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length,nside=nside)


    pixel_edges = np.arange(3*12*nside**2 , dtype=int) 

    maps, offsets = Destriper.run_destriper(pointing,
                                    tod,
                                    weights,
                                    offset_length,
                                    pixel_edges,
                                    obsid,
                                    special_weight=[special_weights,special_weights_rot])

    if rank == 0:
        #write_map_healpix('mcal1_',maps,'maps/cbass/')  
        write_map_iqu_healpix('mcal1_iqu',maps,'maps/cbass/',nside=nside)  

    if rank == 0:
        filelist = tqdm(filelist[lo:hi])
    else:
        filelist = filelist[lo:hi]
    #for i,filename in enumerate(filelist):

    #    select =  (obsid == cbass_data[i].obsid)
    #    cbass_data[i].extra_data['offset'] = offsets['All'][select]
    #
    #    cbass_data[i].save_to_hdf5('cbass_hdf5/mcal1/{}.h5'.format(os.path.basename(filename).split('.fits')[0])) 
