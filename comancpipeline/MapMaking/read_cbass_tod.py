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
import sys 

import psutil 
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Recursively expand slist's objects
# into olist, using seen to track
# already processed objects.
import gc
def _getr(slist, olist, seen):
  for e in slist:
    if id(e) in seen:
      continue
    seen[id(e)] = None
    olist.append(e)
    tl = gc.get_referents(e)
    if tl:
      _getr(tl, olist, seen)

# The public function.
def get_all_objects():
  """Return a list of all live Python
  objects, not including the list itself."""
  gcl = gc.get_objects()
  olist = []
  seen = {}
  # Just in case:
  seen[id(gcl)] = None
  seen[id(olist)] = None
  seen[id(seen)] = None
  # _getr does the real work.
  _getr(gcl, olist, seen)
  return olist


def write_map_iqu_healpix(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        hdulist = []
        m = np.zeros((9,12*nside**2)) + hp.UNSEEN
        npix = 12*nside**2
        m[0,:] = v['I']
        m[1,:] = v['Q']
        m[2,:] = v['U']
        m[3,:] = v['Iw']
        m[4,:] = v['Qw']
        m[5,:] = v['Uw']
        m[6,:] = v['I_rms']
        m[7,:] = v['Q_rms']
        m[8,:] = v['U_rms']
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

def main():
    datadir = '/scratch/nas_cbassarc/cbass_data/Reductions/v34m3/v34m3_gsS_tod/mcal1/'
    filename = f'{datadir}/24-Mar-2014:12:40:39_reload_mcal1.fits'

    filelist = np.loadtxt('filelists/local_AWR1_xND12_xAS14.txt',dtype=str)
    filelist = filelist[:250]
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
    
    nside = 512
    offset_length = 500  # 100 samples = 1 second 
    #tod, weights, pointing, obsid, cbass_data, special_weights = read_comap_data(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length)

    tod, weights, pointing, obsid,special_weights, special_weights_rot,special_weights_pa, cbass_data = read_comap_data_iqu(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length,nside=nside)

    pixel_edges = np.arange(3*12*nside**2 , dtype=int) 

    print('STARTING DESTRIPER')
    maps, offsets = Destriper.run_destriper(pointing,
                                    tod,
                                    weights,
                                    offset_length,
                                    pixel_edges,
                                    obsid,
                                    special_weight=[special_weights,special_weights_rot,special_weights_pa])
    


    if rank == 0:
        write_map_iqu_healpix('mcal1_iqu_final_cut_1000',maps,'maps/cbass/',nside=nside)  

    if rank == 0:
        filelist = tqdm(filelist[lo:hi])
    else:
        filelist = filelist[lo:hi]

    for i,filename in enumerate(filelist):

       select =  (obsid == cbass_data[i].obsid)
       this_offsets = offsets['All'][select]
       cbass_data[i].extra_data['offset'] = np.reshape(this_offsets,(3,-1)) 
       cbass_data[i].extra_data['offset'] = cbass_data[i].extra_data['offset'][:,:]
       cbass_data[i].extra_data['filtered_tod'] = np.reshape(tod[select],(3,-1))

       cbass_data[i].save_to_hdf5('cbass_hdf5/mcal1/{}.h5'.format(os.path.basename(filename).split('.fits')[0])) 

if __name__ == "__main__":
    main()