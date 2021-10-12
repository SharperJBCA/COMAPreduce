import numpy as np
import comancpipeline
import pathlib
import h5py
from skimage import filters

def get_nobs(obsid):
    minobsid = np.min(obsid)
    maxobsid = np.max(obsid)
    count = int(maxobsid)-int(minobsid) + 1
    return count

def smooth_gains(obsid, gains):
    """
    Smooth the calibration functions over all time to get the best signal-to-noise
    """    
    minobsid = np.min(obsid)
    maxobsid = np.max(obsid)
    count = int(maxobsid)-int(minobsid) + 1
    allobs = np.zeros(count)
    allgain= np.zeros(count)
    idx = (obsid-minobsid).astype(int)

    allobs[idx] = obsid
    allgain[idx]= gains
    idx_sort = np.argsort(allobs)
    allobs = np.arange(minobsid,maxobsid+1)
    allgain= allgain[idx_sort]
    bd = (allgain == 0)
    allgain[bd]= np.nan
    gd = np.isfinite(allgain)
    try:
        allgain[~gd] = np.interp(allobs[~gd],allobs[gd],allgain[gd])
        allgain = filters.median(allgain,filters.window('boxcar',101))
    except ValueError:
        pass

    return allobs, allgain

path = pathlib.Path(comancpipeline.__file__).resolve().parent
average_beam_widths = np.loadtxt(f'{path}/data/AverageBeamWidths.dat',skiprows=1)

average_beam_widths = {int(d[0]):d[1:] for d in average_beam_widths}


feed_positions = np.loadtxt(f'{path}/data/COMAP_FEEDS.dat')

feed_positions = {int(d[0]):d[1:] for d in feed_positions}


feed_gains_hd5 = h5py.File(f'{path}/data/gains.hd5','r')

feed_gains = {k:{'obsids':v['obsids'][...],'gains':v['gains'][...]} for k,v in feed_gains_hd5.items()}

for cal_source, data in feed_gains.items():
    obscount = get_nobs(data['obsids'])
    nobs, nfeeds, nroach, nchan = data['gains'].shape
    allgains = np.zeros((obscount, nfeeds, nroach, nchan))

    for ifeed in range(nfeeds):
        for iroach in range(nroach):
            for ichan in range(nchan):
                allobs, allgains[:,ifeed,iroach,ichan] = smooth_gains(data['obsids'],data['gains'][:,ifeed,iroach,ichan])
    
    feed_gains[cal_source]['obsids'] = np.array(allobs).astype(int)
    feed_gains[cal_source]['gains'] = allgains
    feed_gains[cal_source]['frequency'] = np.array([[27.5, 26.5],
                                                    [28.5, 29.5],
                                                    [31.5, 30.5],
                                                    [32.5, 33.5]])

feed_gains_hd5.close()
