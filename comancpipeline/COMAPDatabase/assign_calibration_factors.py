import numpy as np
from matplotlib import pyplot
import h5py
from tqdm import tqdm
import os
import pickle
def main(filename: str,source='TauA'):
    h = h5py.File(filename,'r')
    dv = 2./64
    frequency = np.concatenate(((np.arange(64) + 0.5)*-dv + 28,
                           (np.arange(64) + 0.5)*dv + 28,
                           (np.arange(64) + 0.5)*-dv + 32,
                           (np.arange(64) + 0.5)*dv + 32))
    obsid = list(h.keys())
    obsid = np.array(obsid).astype(int)
    obsididx = np.argsort(obsid)
    obsid = obsid[obsididx]
    obsid = obsid
    tsys = np.zeros((len(obsid), 20, 4, 1024))
    spikes = np.zeros((len(obsid), 20, 4, 1024))
    dv = 2./1024 * 16
    nfreq = 64
    freq = np.array([(np.arange(nfreq)+0.5)*-dv + 28,
                     (np.arange(nfreq)+0.5)* dv + 28,
                     (np.arange(nfreq)+0.5)*-dv + 32,
                     (np.arange(nfreq)+0.5)* dv + 32])

    gain_averages = np.zeros((len(obsid), 20, 4))
    gain_std = np.zeros((len(obsid), 20, 4))
    gain_range = np.zeros((len(obsid), 20, 4))
    all_fnoise = np.zeros((len(obsid), 20, 4,3))
    all_chi2 = np.zeros((len(obsid), 20, 4))

    taua_obs = np.zeros(len(obsid),dtype=bool)
    for i,obs in enumerate(tqdm(obsid)):
        try:
        #if True:
            if not 'TauA' in h[str(obs)]['level2'].attrs['source']:
                continue
            feeds = np.sort([int(f[:-1]) for f in h[str(obs)]['level2'].attrs['pixels'].split() if 'A' in f])
            if (len(feeds) != h[str(obs)]['Vane']['Tsys'].shape[1]):
                feeds = np.append(feeds,20)
            gains=h[str(obs)]['FitSource/Gains'][...]
            if np.nansum(gains) == 0:
                continue
            fnoise = h[str(obs)]['FnoiseStats/fnoise_fits'][...]
            fnoise[fnoise == 0] = np.nan
            chi2 = h[str(obs)]['FitSource/Chi2'][...]
            chi2[chi2 == 0] = np.nan
            freq_mask = [None,[29.5,30],None,[33.5,34]]

            level2_mask = h[str(obs)]['Vane/Level2Mask'][...] 
            for iband in range(4):
                g = gains[:,iband]
                g[g == 0] =np.nan
                f = freq[iband].flatten()
                bad = level2_mask[:,iband]
                if not isinstance(freq_mask[iband],type(None)):
                    bad = bad | ((f[None,:] > freq_mask[iband][0]) & (f[None,:] <= freq_mask[iband][1]))
                g[bad[feeds-1]] = np.nan
                gain_averages[i,feeds-1,iband] = np.nanmean(g,axis=-1)
                gain_std[i,feeds-1,iband]   = np.nanstd(g,axis=-1)
                gain_range[i,feeds-1,iband] = np.nanmax(g,axis=-1)-np.nanmin(g,axis=-1)
                
                all_fnoise[i,feeds-1,iband,:] = np.nanmean(fnoise[0,iband,:,0,:])
                all_chi2[i,feeds-1,iband]   = np.nanmean(chi2[0,iband,:,1])

            taua_obs[i] = True
        except KeyError:
           tsys[i,feeds-1,:,:]   = np.nan
           spikes[i,feeds-1,:,:] = np.nan
           print(obs)
           continue
    h.close()

    taua_mask   = np.zeros((len(np.where(taua_obs)[0]),20,4),dtype=bool)
    taua_values = np.zeros((len(np.where(taua_obs)[0]),20,4))
    for ifeed in range(19):
        cuts = np.loadtxt('datecuts/Feed{:02d}_cuts.dat'.format(ifeed+1),dtype=float,usecols=[0,1])
        for iband in range(4):
            y = gain_averages[taua_obs,ifeed,iband]
            e = gain_std[taua_obs,ifeed,iband]
            o = obsid[taua_obs]
            gd = np.ones(y.size,dtype=bool)
            for icut,(start,end) in enumerate(cuts):
                lo = np.argmin((start-o)**2)
                hi = np.argmin((end-o)**2)
                gd[lo:hi+1] = np.abs(y[lo:hi+1]-np.nanmedian(y[lo:hi+1])) < 0.05
                pyplot.axvspan(start,end,alpha=0.15,color='C{}'.format(icut))

            red   = all_fnoise[taua_obs,ifeed,iband,0]
            alpha = all_fnoise[taua_obs,ifeed,iband,1]
            gd    = gd & (alpha > -1.1) & (red < 5e-3) & ((y/e) > 5 ) & (e < 0.05)
            taua_mask[:,ifeed,iband] = gd
            taua_values[:,ifeed,iband] = y*1
            
            pyplot.errorbar(o[gd],
                            y[gd],
                            fmt='.',yerr=e[gd])
            #for icut,(start,end) in enumerate(cuts):
            #    pyplot.axvspan(start,end,alpha=0.15,color='C{}'.format(icut))
        pyplot.ylim(0.5,1.)
        pyplot.xlabel('Obs ID')
        pyplot.ylabel('Gain')
        pyplot.title(ifeed+1)
        pyplot.savefig('figures/gains/{:02d}_taua_gains.png'.format(ifeed+1))
        pyplot.clf()

    # Remove factors that aren't good in the three central feeds and all 4 bands
    time_taua_mask = (np.sum(taua_mask[:,:3,:],axis=(1,2)) == 12)
    taua_mask[~time_taua_mask] = False
    taua_values = taua_values[time_taua_mask,...] # good gain factors
    taua_obsids = (obsid[taua_obs])[time_taua_mask] # taua_obsids

    # Now apply the calibration factors

    ## Write the gain factors per feed to the database
    ## Creates a new group called "level3" that is used
    ## by the level3 creation routines.
    h = h5py.File(filename,'a')
    nFeeds = 19
    for obs in obsid:
        gain = np.zeros((nFeeds,4))
        for ifeed in range(nFeeds):
            cuts = np.loadtxt('datecuts/Feed{:02d}_cuts.dat'.format(ifeed+1),dtype=float,usecols=[0,1])
            for icut,(start,end) in enumerate(cuts):
                if (start <= obs < end):
                    hi = np.argmin((taua_obsids - end)**2)
                    lo = np.argmin((taua_obsids - start)**2)
                    o = taua_obsids[lo:hi+1]
                    y = taua_values[lo:hi+1,ifeed]
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
