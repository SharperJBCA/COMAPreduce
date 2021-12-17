import numpy as np
import comancpipeline
import pathlib
import h5py
from skimage import filters
from comancpipeline.Tools import stats
def get_nobs(obsid):
    minobsid = np.min(obsid)
    maxobsid = np.max(obsid)
    count = int(maxobsid)-int(minobsid) + 1
    return count

def smooth_gains(obsid, gains,errs):
    """
    Smooth the calibration functions over all time to get the best signal-to-noise
    """    
    minobsid = np.min(obsid)
    maxobsid = np.max(obsid)
    count = int(maxobsid)-int(minobsid) + 1
    allobs = np.zeros(count)
    allerrs= np.zeros(count)
    allgain= np.zeros(count)
    idx = (obsid-minobsid).astype(int)

    allobs[idx] = obsid
    allgain[idx]= gains
    allerrs[idx]= errs
    idx_sort = np.argsort(allobs)
    #allobs = np.arange(minobsid,maxobsid+1)
    allgain= allgain[idx_sort]
    allobs = allobs[idx_sort]
    allerrs= allerrs[idx_sort]
    med = np.nanmedian(gains)
    bd = (allgain < 0.5) | (allgain > 1) | (np.abs(allgain-med) > 2) | (allerrs > 1e-3)
    allgain[bd]= np.nan
    gd = np.isfinite(allgain)

    #pyplot.errorbar(allobs[gd],allgain[gd],fmt='.',yerr=allerrs[gd],capsize=3)
    try:
        allgain[~gd] = np.interp(allobs[~gd],allobs[gd],allgain[gd])
        allgain_mdl = filters.median(allgain,filters.window('boxcar',51))
    except ValueError:
        return allobs, allgain
    
    rms=  stats.MAD(allgain[gd]-allgain_mdl[gd])
    bd = (allgain < 0.5) | (allgain > 1) | (np.abs(allgain-med) > 2) | (allerrs > 1e-3) | (np.abs(allgain-allgain_mdl) > 3*rms)
    allgain[bd]= np.nan
    gd = np.isfinite(allgain)
    try:
        allgain[~gd] = np.interp(allobs[~gd],allobs[gd],allgain[gd])
        allgain = filters.median(allgain,filters.window('boxcar',51))
    except ValueError:
        pass

    return allobs, allgain

def read_gains():

    path = pathlib.Path(comancpipeline.__file__).resolve().parent
    average_beam_widths = np.loadtxt(f'{path}/data/AverageBeamWidths.dat',skiprows=1)

    average_beam_widths = {int(d[0]):d[1:] for d in average_beam_widths}


    feed_positions = np.loadtxt(f'{path}/data/COMAP_FEEDS.dat')
    
    feed_positions = {int(d[0]):d[1:] for d in feed_positions}


    feed_gains_hd5 = h5py.File(f'{path}/data/gains.hd5','r')

    feed_gains = {k:{'obsids':v['obsids'][...],'gains':v['gains'][...],'errors':v['errors'][...]} for k,v in feed_gains_hd5.items()}

    for cal_source, data in feed_gains.items():
        obscount = get_nobs(data['obsids'])
        nobs, nfeeds, nchan = data['gains'].shape
        allgains = np.zeros((obscount, nfeeds, nchan))
        for ifeed in range(nfeeds):
            for ichan in range(nchan):
                allobs, allgains[:,ifeed,ichan] = smooth_gains(data['obsids'],
                                                               data['gains'][:,ifeed,ichan],
                                                               data['errors'][:,ifeed,ichan])
    
        feed_gains[cal_source]['obsids'] = np.array(allobs).astype(int)
        feed_gains[cal_source]['gains'] = allgains
        feed_gains[cal_source]['frequency'] = np.array([[27.5, 26.5],
                                                        [28.5, 29.5],
                                                        [31.5, 30.5],
                                                        [32.5, 33.5]])

    feed_gains_hd5.close()

    return feed_positions, feed_gains

try:
    feed_positions, feed_gains =read_gains()
except KeyError:
    pass
