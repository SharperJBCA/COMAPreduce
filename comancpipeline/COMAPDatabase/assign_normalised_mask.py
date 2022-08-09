# assign_normalised_mask
#
# "Normalised" channel masks that are applied uniformly over date ranges
# defined in datecuts/ 
# These masks are used in conjunction with the initial Tsys flags and 
# are applied at the level 3 creation stage.


import numpy as np
from matplotlib import pyplot
import h5py
from tqdm import tqdm
import os
import pickle
from scipy.ndimage import gaussian_filter1d
from comancpipeline.COMAPDatabase.common_functions import *
def get_data(h, nfeeds=20, 
             nbands_level1=4, nfreq_level1=1024,
             nfreq_level2=64,obsid=0):
    """
    Read the database data into a dictionary

    h = h[obsid]
    """
    freq_mask = [None,[29.5,30],None,[33.5,34]]

    output_data = {}

    if not all([v in h for v in ['level2','Vane']]):
        return output_data
    if not all([v in h['Vane'] for v in ['Spikes']]):
        return output_data

    if not all([v in h['level2'].attrs for v in ['source','pixels']]):
        return output_data

    if not all([len(h[k].shape) == 4 for k in ['Vane/Spikes','Vane/Tsys']]):
        return output_data

    feeds = get_feeds(h)

    out_keys = ['tsys','spikes']
    keys = ['Vane/Tsys','Vane/Spikes']
    for outk, k in zip(out_keys, keys):
        dshape = list(h[k].shape)
        dtype = h[k].dtype
        if len(dshape) == 4:
            dshape[1] = nfeeds
        else:
            dshape[0] = nfeeds
        output_data[outk] = np.zeros(dshape,dtype=dtype)
        output_data[outk][:,feeds-1,...] = h[k][...]

    output_data['good'] = np.array([True])
    return output_data

def assign_normalised_mask(filename,dv=2./1024.,nfeeds=20,
                           nbands_level1=4, nfreq_level1=1024):

    file_dir = os.path.dirname(__file__) 
    h = h5py.File(filename,'r')

    obsid = get_allkeys(h)

    output_data = {'tsys'       : np.zeros((len(obsid), 2, nfeeds, nbands_level1, nfreq_level1)),
                   'spikes'     : np.zeros((len(obsid), 2, nfeeds, nbands_level1, nfreq_level1)),
                   'good'       : np.zeros((len(obsid)),dtype=bool),
                   'mask'       : np.zeros((len(obsid), nfeeds, nbands_level1, nfreq_level1))}


    frequency = np.array([(np.arange(1024)+0.5)*-dv + 28,
                          (np.arange(1024)+0.5)*dv  + 28,
                          (np.arange(1024)+0.5)*-dv + 32,
                          (np.arange(1024)+0.5)*dv  + 32])
    for i,obs in enumerate(tqdm(obsid)):
        #try:
            data = get_data(h[str(obs)],obsid=obs)
            for k,v in data.items():
                output_data[k][i] = v

        #except KeyError:
        #    tsys[i,feeds-1,:,:]   = np.nan
        #    spikes[i,feeds-1,:,:] = np.nan
        #    print(obs)
        #    continue
    h.close()

    # Create the normalised masks
    for ifeed in range(0,19):
        # [icut,start/end]
        cuts = np.loadtxt('{}/datecuts/Feed{:02d}_cuts.dat'.format(file_dir,ifeed+1),dtype=float,usecols=[0,1])
        for iband in range(4):            
            for (start,end) in cuts:
                lo = np.argmin((obsid - start)**2)
                hi = np.argmin((obsid - end)**2)
                s = np.nansum(output_data['spikes'][lo:hi+1,0,ifeed,iband,:],axis=0)
                w = np.sum(np.isfinite(output_data['spikes'][lo:hi+1,0,ifeed,iband,:]),axis=0)
                sel = np.where((s>0.25*w))[0]
                output_data['mask'][lo:hi+1,ifeed,iband,sel] = 1


    h = h5py.File(filename,'a')

    Nbin = 16
    for iobs,obs in enumerate(obsid):
        if not output_data['good'][iobs]:
            continue

        m = 1-output_data['mask'][iobs]
        s = 1-output_data['spikes'][iobs,0]
        
        m = np.sum(np.reshape(m,(20, 4, 1024//16, 16)),axis=-1)
        s = np.sum(np.reshape(s,(20, 4, 1024//16, 16)),axis=-1)

        level2_mask = (m < (Nbin - 1)) | (s < (Nbin -1))
        level2_mask = level2_mask | np.roll(level2_mask,-1) | np.roll(level2_mask,1)

        grp = h[str(obs)]['Vane']
        if 'Level2Mask' in grp:
            del grp['Level2Mask']
        grp.create_dataset('Level2Mask',data=level2_mask)
    h.close()

if __name__ == "__main__":

    filename = 'comap_database.hdf5'
    main(filename)
