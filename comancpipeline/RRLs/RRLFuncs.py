import numpy as np
import h5py 

from comancpipeline.MapMaking import MapTypes
import sys
from tqdm import tqdm
from astropy.io import fits
from matplotlib import pyplot


import numpy as np
from matplotlib import pyplot
from comancpipeline.MapMaking import MapTypes
from comancpipeline.Analysis import Statistics,BaseClasses
import h5py
import sys
from tqdm import tqdm
import os

class RepointEdges(BaseClasses.DataStructure):
    """
    Scan Edge Split - Each time the telescope stops to repoint this is defined as the edge of a scan
    """

    def __init__(self, **kwargs):

        self.max_el_current_fraction = 0.7
        self.min_sample_distance = 10
        self.min_scan_length = 5000 # samples
        self.offset_length = 50
        for item, value in kwargs.items():
            self.__setattr__(item,value)

    def __call__(self, data):
        """
        Expects a level 2 data structure
        """
        return self.getScanPositions(data)

    def getScanPositions(self, d):
        """
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,
        provides indices for the positions of scans in masked array

        Notes:
        - We may need to check for vane position too
        - Iteratively finding the best current fraction may also be needed
        """
        features = d['level1/spectrometer/features'][:]
        uf, counts = np.unique(features,return_counts=True) # select most common feature
        ifeature = np.floor(np.log10(uf[np.argmax(counts)])/np.log10(2))
        selectFeature = self.featureBits(features.astype(float), ifeature)
        index_feature = np.where(selectFeature)[0]

        # make it so that you have a gap, only use data where the telescope is moving

        # Elevation current seems a good proxy for finding repointing times
        elcurrent = np.abs(d['level1/hk/antenna0/driveNode/elDacOutput'][:])
        elutc = d['level1/hk/antenna0/driveNode/utc'][:]
        mjd = d['level1/spectrometer/MJD'][:]

        # these are when the telescope is changing position
        select = np.where((np.gradient(elcurrent) > 1000))[0]#np.max(elcurrent)*self.max_el_current_fraction))[0]

        dselect = select[1:]-select[:-1]
        large_step_indices = np.where((dselect > self.min_sample_distance))[0]

        ends = select[np.append(large_step_indices,len(dselect)-1)]

        # Now map these indices to the original indices
        scan_edges = []
        for (start,end) in zip(ends[:-1],ends[1:]):
            tstart,tend = np.argmin((mjd-elutc[start])**2),np.argmin((mjd-elutc[end])**2)

            # Need to check we are not in a bad feature region
            if selectFeature[tstart] == 0:
                tstart = index_feature[np.argmin((index_feature - tstart)**2)]
            if selectFeature[tend] == 0:
                tend = index_feature[np.argmin((index_feature - tend)**2)]

            if (tend-tstart) > self.min_scan_length:
                Nsteps = int((tend-tstart)//self.offset_length)
                scan_edges += [[tstart,tstart+self.offset_length*Nsteps]]

        return scan_edges

edge_obj =  RepointEdges()

from scipy.optimize import minimize
def model(P,az,el):
    return P[0]/np.sin(el) + P[1]*az + P[2]

def error(P,z,az,el):
    return np.sum((model(P,az,el)-z)**2)

def fit_atmos(z,az,el):
    """
    Fit for a simple atmospheric slab model + gradients in azimuth

    T_atm = A*(azimuth - mean(azimuth)) + B/sin(elevation) + C
    """

    P0 = [np.nanstd(z), np.nanstd(z), np.nanmedian(z)]#,0,0]
    daz = az - np.nanmedian(az)
    daz[daz > 180] -= 360
    result = minimize(error,P0,args=(z,(az-np.nanmedian(az))*np.pi/180.,el*np.pi/180.))

    return model(result.x,(az-np.nanmedian(az))*np.pi/180.,el*np.pi/180.)
    
def fit_gradients(veloc,z,outer):
    """
    Fit for the offsets across the band. 
    
    Fit is performed using a LA implementation of least-squares.
    """
    #outer = (np.abs(veloc) > 120)
    N = int(np.sum(outer))
    F = np.ones((2,N))
    # Solve for the outer part of the spectrum
    p =  np.arange(z.shape[-1])[outer]

    F[0,:] = np.arange(z.shape[-1])[outer]
    C = np.linalg.inv(F.dot(F.T))
    b = np.sum(F[None,:,:]*z[:,None,outer],axis=2)
    a = np.sum(C[None,:,:]*b[:,:,None],axis=1)

    # Solve for the full spectrum
    F = np.ones((2,z.shape[-1]))
    F[0,:] = np.arange(z.shape[-1])
    f = np.sum(F[None,:,:]*a[:,:,None],axis=1)
    return f


def read_data(filename,mbin,feedidxs = np.array([1,2,3,8,9,10,11,12,13,14,15,16,17,18,19]).astype(int),veloc_upper=120,veloc_lower=-50):
    """
    Read in all the data from a given list of files

    Arguments
    filelistname - List of strings containing the files to processed
    mbin - The MapTypes.FlatMapType object you will be using to create the data cube
     
    Keywords
    feedidxs - List of integers indexing the COMAP feed assignments (1...19)
    """

    nfeeds = len(feedidxs)


    h = h5py.File(filename,'r')

    # Calculates the scan edges of each observation
    scan_edges = edge_obj(h)

    # The feeds being used for this observation
    feeds = h['level1/spectrometer/feeds'][:]
    idfeeds = []
    for feed in feedidxs:
        if feed in feeds:
            idfeeds += [np.argmin((feeds-feed)**2)]
    ra    = h['level1/spectrometer/pixel_pointing/pixel_ra'][idfeeds,...]
    dec   = h['level1/spectrometer/pixel_pointing/pixel_dec'][idfeeds,...]
    az    = h['level1/spectrometer/pixel_pointing/pixel_az'][idfeeds,...]
    el    = h['level1/spectrometer/pixel_pointing/pixel_el'][idfeeds,...]
    tod   = h['level2rrl/tod'][...]
    veloc = h['level2rrl/velocity'][0,...]

    pixels=mbin[0][0].getFlatPixels(ra,dec)
    nlines = tod.shape[1]
    nfreqs = tod.shape[2]
    sig = np.zeros((len(feedidxs),nlines,nfreqs,tod.shape[-1]))
    wei = np.zeros((len(feedidxs),nlines,nfreqs,tod.shape[-1]))
    edges = np.linspace(-225,225,nlines+1)

    # This loop "cleans" the TOD, removing atmospheric fluctuations in time and baselines in frequency
    weights = np.zeros(tod.shape)
    for ifeed, feed in enumerate(tqdm(feedidxs)):
        if feed == 20:
            continue
        if not feed in feeds:
            continue
        data_feed = np.argmin((feed-feeds)**2)
        for (start,end) in scan_edges:
            for iqno in range(tod.shape[1]):
                # First we select just the data we want
                tod_cut = tod[data_feed,iqno,:,start:end]

                # Now we remove an estimate of the atmospheric slab from each frequency channel.
                for ichan in range(nfreqs):
                    t = fit_atmos(tod_cut[ichan],az[ifeed,start:end],el[ifeed,start:end])
                    tod_cut[ichan] -= t

                # We then fit a baseline using data that is v > v_upper and v < v_lower
                outer = (veloc[iqno] > veloc_upper) | (veloc[iqno] < veloc_lower)
                f = fit_gradients(veloc[iqno],tod_cut.T,outer).T
                tod_cut -= f

                
                N = (end-start)//2*2
                diff = tod[data_feed,iqno,:,start:start+N:2]-tod[data_feed,iqno,:,start+1:start+N:2]
                weights[data_feed,iqno,:,start:end] += 1./np.nanstd(diff,axis=1)[:,None]**2
        
    select = np.concatenate([np.arange(start,end).astype(int) for (start,end) in scan_edges])
    for ifeed, feed in enumerate(tqdm(feedidxs)):
        if feed == 20:
            continue
        if not feed in feeds:
            continue

        # Search for the correct index to the feed in the level2 file
        # We do this because not all observations use all 19 feeds.
        data_feed = np.argmin((feed-feeds)**2)

        for iqno in range(tod.shape[1]):
            gd = np.isfinite(tod[data_feed,iqno])
            if np.sum(gd) == 0:
                continue
            tod[data_feed,iqno,~gd] = 0
            gd = gd.astype(float)
            sig[ifeed,iqno] += tod[data_feed,iqno]*weights[data_feed,iqno]*gd
            wei[ifeed,iqno] += weights[data_feed,iqno]*gd
                        
    for iqno in range(sig.shape[1]):
        for ifreq in range(sig.shape[2]):
            mbin[iqno][ifreq].sum_data(sig[:,iqno,ifreq,select].flatten(),pixels[:,select].flatten(),weights=wei[:,iqno,ifreq,select].flatten())
    h.close()
