# fit_cali_maps
# 
# Fit the amplitude and position of source in calibration maps

import numpy as np
from matplotlib import pyplot
import glob
import h5py
from tqdm import tqdm
from astropy.wcs import WCS
import os
import sys
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs
from comancpipeline.Tools import WCS as myWCS
from astropy.modeling import models, fitting

def fit_source_emcee(tod,x,y):
    """

    args:
    tod -- ndarray (float) [N_tod] :
    x   -- ndarray (float) [N_tod] :
    y   -- ndarray (float) [N_tod] : 

    """
    fit = fitting.LevMarLSQFitter()
    
    m_init = models.Gaussian2D(amplitude=np.max(tod)-np.median(tod),
                               x_mean=0,
                               y_mean=0,
                               x_stddev=4./60./2.355,
                               y_stddev=4./60./2.355,
                               theta=0)
    m_init += models.Const2D(amplitude=np.median(tod))
    model_fit = fit(m_init, x, y, tod)
    output = [getattr(model_fit,name).value for name in model_fit.param_names]

    return output, model_fit.param_names

def fit_source(tod,x,y):
    """

    args:
    tod -- ndarray (float) [N_tod] :
    x   -- ndarray (float) [N_tod] :
    y   -- ndarray (float) [N_tod] : 

    """
    fit = fitting.LevMarLSQFitter()
    
    m_init = models.Gaussian2D(amplitude=np.max(tod)-np.median(tod),
                               x_mean=0,
                               y_mean=0,
                               x_stddev=4./60./2.355,
                               y_stddev=4./60./2.355,
                               theta=0)
    m_init += models.Const2D(amplitude=np.median(tod))
    model_fit = fit(m_init, x, y, tod)
    output = [getattr(model_fit,name).value for name in model_fit.param_names]

    return output, model_fit.param_names

def read_file(filename):
    """
    """
    N_FEEDS = 20
    N_BANDS = 4

    fit_vals = np.zeros((7,N_FEEDS, N_BANDS))
    h = fits.open(filename,memmap=False)
    wcs = WCS(h[0].header)
    wcs = wcs.dropaxis(2)

    pix_coords = myWCS.get_pixel_coordinates(wcs,h[0].data[0].shape)
    pix_coords[0][pix_coords[0] > 180] -= 360
    r = np.sqrt(pix_coords[0]**2 + pix_coords[1]**2)
    param_names = []
    for ifeed in range(N_FEEDS):
        for iband in range(N_BANDS):
            mflat = h[ifeed].data[iband,...].flatten()
            gd = (r < 10./60.) & (mflat != 0) & np.isfinite(mflat)
            if np.sum(mflat[gd]) == 0:
                continue
            fit_vals[:,ifeed,iband], param_names = fit_source(mflat[gd],pix_coords[0][gd], pix_coords[1][gd])
            #fit_vals[:,ifeed,iband], param_names = fit_source_emcee(mflat[gd],pix_coords[0][gd], pix_coords[1][gd],fit_vals[:,ifeed,iband])
    h.close()

    return fit_vals, param_names

def save_to_level3(filename, fit_vals, param_names):

    h = h5py.File(filename,'a')

    if 'FittedFluxes' in h:
        del h['FittedFluxes']

    grp = h.create_group('FittedFluxes')

    for k, v in zip(param_names, fit_vals):
        grp.create_dataset(k,data=v)

    h.close()
    

if __name__ == "__main__":


    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
   
    # get file list
    filelist = np.array(glob.glob('TauA/pointing_corrected_fits_maps/*.fits'))
    obsids_fits = np.array([int(os.path.basename(f).split('_')[0]) for f in filelist])

    level3_filelist = glob.glob('mock_level3_TauA_update2/level3_*')
    obsids_lvl3 = [int(os.path.basename(f).split('-')[1]) for f in level3_filelist]
    level3_dict = {obsid:f for obsid,f in zip(obsids_lvl3,level3_filelist)}

    N_files = len(filelist)
    idx = (np.sort(np.mod(np.arange(N_files),size)) == rank)


    if rank == 0:
        floop = tqdm(filelist[idx])
    else:
        floop = filelist[idx]
    # fit each feed/band in maps
    for obsid, filename in zip(obsids_fits[idx], floop):

        fit_vals, param_names = read_file(filename)

        # save information to level 3 file
        save_to_level3(level3_dict[obsid], fit_vals, param_names)
