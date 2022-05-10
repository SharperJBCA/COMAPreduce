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

def main(filename):
    h = h5py.File(filename,'r')

    obsid = list(h.keys())
    obsid = np.array(obsid).astype(int)
    obsididx = np.argsort(obsid)
    obsid = obsid[obsididx]

    tsys = np.zeros((len(obsid), 20, 4, 1024))
    spikes = np.zeros((len(obsid), 20, 4, 1024))
    mask = np.zeros((len(obsid), 20, 4, 1024))
    dv = 2./1024
    freq = np.array([(np.arange(1024)+0.5)*-dv + 28,
                     (np.arange(1024)+0.5)*dv  + 28,
                     (np.arange(1024)+0.5)*-dv + 32,
                     (np.arange(1024)+0.5)*dv  + 32])
    for i,obs in enumerate(tqdm(obsid)):
        try:
            feeds = np.sort([int(f[:-1]) for f in h[str(obs)]['level2'].attrs['pixels'].split() if 'A' in f])
            if (len(feeds) != h[str(obs)]['Vane']['Tsys'].shape[1]):
                feeds = np.append(feeds,20)
            try:
                tsys[i,feeds-1,:,:] = h[str(obs)]['Vane']['Tsys'][0,:,:,:]
                spikes[i,feeds-1,:,:] = h[str(obs)]['Vane']['Spikes'][0,...]
            except ValueError:
                print(feeds)
                print(tsys[i,feeds-1,:,:].shape)
            
        except KeyError:
            tsys[i,feeds-1,:,:]   = np.nan
            spikes[i,feeds-1,:,:] = np.nan
            print(obs)
            continue
    h.close()

    # Create the normalised masks
    for ifeed in range(0,19):
        # [icut,start/end]
        cuts = np.loadtxt('datecuts/Feed{:02d}_cuts.dat'.format(ifeed+1),dtype=float,usecols=[0,1])
        fig = pyplot.figure()
        for iband in range(4):
            ax=pyplot.subplot(221+iband)
            
            for (start,end) in cuts:
                lo = np.argmin((obsid - start)**2)
                hi = np.argmin((obsid - end)**2)
                s = np.nansum(spikes[lo:hi+1,ifeed,iband,:],axis=0)
                w = np.sum(np.isfinite(spikes[lo:hi+1,ifeed,iband,:]),axis=0)
                sel = np.where((s>0.25*w))[0]
                mask[lo:hi+1,ifeed,iband,sel] = 1
            pyplot.imshow(mask[:,ifeed,iband,:],aspect='auto',interpolation='none')

        pyplot.suptitle(ifeed+1)
        pyplot.savefig('figures/tsys/{:02d}_tsys_normalised_mask.png'.format(ifeed+1))
        pyplot.close(fig)

    h = h5py.File(filename,'a')

    Nbin = 16
    for iobs,obs in enumerate(obsid):
        if not 'Vane' in h[str(obs)]:
            continue
        m = 1-mask[iobs]
        s = 1-spikes[iobs]
        
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
