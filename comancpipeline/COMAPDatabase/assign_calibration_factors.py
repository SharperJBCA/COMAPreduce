import numpy as np
from matplotlib import pyplot
import h5py
from tqdm import tqdm
import os
import pickle

from comancpipeline.COMAPDatabase.common_functions import *

def get_data(h, output_data, source, nfeeds=20, 
             nbands_level1=4, nfreq_level1=1024,
             nfreq_level2=64,obsid=0):
    """
    Read the database data into a dictionary

    h = h[obsid]
    """
    freq_mask = [None,[29.5,30],None,[33.5,34]]

    output_data = {}

    if not all([v in h for v in ['level2','Vane','FitSource']]):
        return output_data
    if not all([v in h['level2'].attrs for v in ['source','pixels']]):
        return output_data
    if not all([v in h['Vane'] for v in ['Level2Mask']]):
        return output_data

    if not source in h['level2'].attrs['source']:
        return output_data

    feeds = get_feeds(h)

    if np.nansum(h['FitSource/Gains'][...]) == 0:
        return output_data

    out_keys = ['gains','fnoise','chi2']
    keys = ['FitSource/Gains','FnoiseStats/fnoise_fits','FitSource/Chi2']

    # We need to calculate the rednoise at 0.3Hz
    fstats = h['FnoiseStats']
    ps = fstats['powerspectra'][...,0,:]
    nu = fstats['freqspectra'][...,0,:]
    v = fstats['fnoise_fits'][...]
    rednoise = v[...,0,0]*(0.3/2.)**v[...,0,1]
    model = v[:,:,:,0,2,None] + v[:,:,:,0,0,None]*(nu/2.)**v[:,:,:,0,1,None]
    residual = ps-model
    residual_sum = np.sqrt(np.nansum(residual**2,axis=-1))
    residual_sum[residual_sum==0]=np.nan
            

    dshape = list(rednoise.shape)
    dtype = rednoise.dtype
    dshape[0] = nfeeds
    output_data['rednoise'] = np.zeros(dshape,dtype=dtype)
    output_data['rednoise'][feeds-1] = rednoise


    dshape = list(residual_sum.shape)
    dtype = rednoise.dtype
    dshape[0] = nfeeds
    output_data['residual'] = np.zeros(dshape,dtype=dtype)
    output_data['residual'][feeds-1] = residual_sum

    for outk, k in zip(out_keys, keys):
        dshape = list(h[k].shape)
        dtype = h[k].dtype
        dshape[0] = nfeeds
        output_data[outk] = np.zeros(dshape,dtype=dtype)
        output_data[outk][feeds-1] = h[k][...]

    output_data['level2_mask'] = h['Vane/Level2Mask'][...]
    output_data['cal_obs'] = np.array([True])

    output_data['cal_good'] = np.zeros((nfeeds,nbands_level1), dtype=bool)
    for ifeed in range(nfeeds):
        for iband in range(nbands_level1):
            gains_feed0 = output_data['gains'][ifeed,iband,:]
            gains_mask = (gains_feed0 > 0.1) & (gains_feed0 < 1) # Remove the most obvious bad fits
            gains_flag = (np.sum(gains_mask) > gains_mask.size*0.5)
            if gains_flag:
                output_data['cal_good'][ifeed,iband]=True

    # Check for bad weather
    if (np.nanmedian(rednoise[0,0]) > 0.15) |\
       (np.nanmedian(rednoise[0,0]) < 0)    |\
       (np.nanmedian(residual_sum[0,0]) > 50):
        output_data['cal_good'][...] = False
    return output_data
    

def assign_calibration_factors(filename: str,
                               source='TauA',
                               dv=32./1024.,
                               nfreq_level2 = 64,
                               nfreq_level1 = 1024,
                               nfeeds = 20,
                               nbands_level1 = 4):
                               
    h = h5py.File(filename,'r')
    file_dir = os.path.dirname(__file__) 

    
    dv = 2./64
    frequency = np.concatenate(((np.arange(nfreq_level2) + 0.5)*-dv + 28,
                                (np.arange(nfreq_level2) + 0.5)*dv + 28,
                                (np.arange(nfreq_level2) + 0.5)*-dv + 32,
                                (np.arange(nfreq_level2) + 0.5)*dv + 32))

    obsid = get_allkeys(h)

    output_data = {'tsys'       : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level1)),
                   'spikes'     : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level1)),
                   'gains'      : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2)),
                   'fnoise'     : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2,1, 3)),
                   'rednoise'   : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2)),
                   'residual'   : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2)),
                   'chi2'       : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2,2)),
                   'level2_mask': np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level2)),
                   'cal_obs': np.zeros(len(obsid),dtype=bool),
                   'cal_good': np.zeros((len(obsid),nfeeds,nbands_level1), dtype=bool),
                   'cal_obsids': np.zeros(len(obsid))}


    # Loop over each obsid and read in data 
    for i,obs in enumerate(tqdm(obsid)):
        data = get_data(h[str(obs)],output_data, source,obsid=obs)
        for k,v in data.items():
            output_data[k][i] = v
        if output_data['cal_obs'][i]:
            output_data['cal_obsids'][i]=obs

    h.close()

    ## Write the gain factors per feed to the database
    ## Creates a new group called "level3" that is used
    ## by the level3 creation routines.
    h = h5py.File(filename,'a')
    for obs in obsid:
        gain = np.zeros((nfeeds,nbands_level1,nfreq_level2))
        for ifeed in range(nfeeds-1):
            cuts = np.loadtxt('{}/datecuts/Feed{:02d}_cuts.dat'.format(file_dir,ifeed+1),dtype=float,usecols=[0,1])
            for icut,(start,end) in enumerate(cuts):
                if (start <= obs < end):
                    hi = np.argmin((output_data['cal_obsids'] - end)**2)
                    lo = np.argmin((output_data['cal_obsids'] - start)**2)
                    o = output_data['cal_obsids'][lo:hi+1]
                    y = output_data['gains'][lo:hi+1,ifeed]
                    select = np.argmin((o - obs)**2)
                    gain[ifeed] = y[select]
        
        if not 'level3' in h[str(obs)]:
            grp = h[str(obs)].create_group('level3')
        else:
            grp = h[str(obs)]['level3']
        if '{}MainBeamFactor'.format(source) in grp:
            del grp['{}MainBeamFactor'.format(source)]
        grp.create_dataset('{}MainBeamFactor'.format(source),data=gain)
    h.close()


if __name__ == "__main__":

    filename = 'comap_database.hdf5'
    main(filename)
