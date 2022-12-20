# update_pointing.py
# SH - 24/11/22
#
# Update the pointing of each observation
# to use the precalculated mean azimuth offset
# A_tele -= dAz/cos(E_tele)

import numpy as np
from matplotlib import pyplot
import glob
import h5py
from tqdm import tqdm
from astropy.wcs import WCS
import os
import sys
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs, CaliModels,UnitConv

def update_pointing(filename):
    """Update the pointing information
    """
    GRP_NAME = 'updated_pixel_pointing'
    N_FEEDS  = 20
    az_off = np.array([ 0.81736919 , 0.88856374 , 0.75226465 , 0.83386315 , 0.78688055 , 0.82400895,
                        1.00569064,  0.90985977,  0.67139243,  0.51411407,  0.64341966,  0.93197356,
                        0.91409291,  0.68841449,  0.63825983,  0.64568386,  0.95573469,  1.24559405,
                        1.15014756, -0.11470404])/60. # Offsets for each feed calculated from Tau A

    h = h5py.File(filename,'a')
    el = h['pixel_pointing']['pixel_el'][...]
    az = h['pixel_pointing']['pixel_az'][...] - az_off[:,None]/np.cos(el*np.pi/180.)
    mjd= h['MJD'][...]

    ra = np.zeros(el.shape)
    dec= np.zeros(el.shape)
    for ifeed in range(N_FEEDS):
        if ifeed == 19:
            continue
        ra[ifeed], dec[ifeed] = Coordinates.h2e_full(az[ifeed], el[ifeed], mjd, Coordinates.comap_longitude, Coordinates.comap_latitude)

    if GRP_NAME in h:
        del h[GRP_NAME]
    grp = h.create_group(GRP_NAME)

    for v,k in zip([az,el,ra,dec],['pixel_az','pixel_el','pixel_ra','pixel_dec']):
        grp.create_dataset(k,data=v)


    h.close()

if __name__ == "__main__":

    LVL3_DIR = 'mock_level3_co2_update2'
    filelist = np.array(glob.glob(f'{LVL3_DIR}/level3_*'))

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    N_files = len(filelist)
    idx = (np.sort(np.mod(np.arange(N_files),size)) == rank)
    
    
    if rank == 0:
        floop = tqdm(filelist[idx])
    else:
        floop = filelist[idx]

    for filename in floop:
        update_pointing(filename)
