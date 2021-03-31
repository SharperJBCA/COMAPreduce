import numpy as np
from matplotlib import pyplot
import h5py
import healpy as hp
import sys
from tqdm import tqdm
from comancpipeline.Tools import Coordinates
from matplotlib.transforms import ScaledTranslation
from scipy.signal import fftconvolve


def MAD(d,axis=0):
    """
    Return Median Absolute Deviation for array along one axis
    """
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48

    return rms

def AutoRMS(tod):
    """
    Auto-differenced RMS
    """
    if len(tod.shape) == 2:
        N = (tod.shape[0]//2)*2
        diff = tod[1:N:2,:] - tod[:N:2,:]
        rms = np.nanstd(diff,axis=0)/np.sqrt(2)
    else:
        N = (tod.size//2)*2
        diff = tod[1:N:2] - tod[:N:2]
        rms = np.nanstd(diff)/np.sqrt(2)

    return rms

def TsysRMS(tod,sample_rate,bandwidth):
    """
    Calculate Tsys from the RMS
    """
    rms =  AutoRMS(tod) 
    Tsys = rms*np.sqrt(bandwidth/sample_rate)
    return Tsys

def weighted_mean(x,e):
    """
    calculate the weighted mean
    """

    return np.sum(x/e**2)/np.sum(1./e**2)

def weighted_var(x,e):
    """
    calculate weighted variance
    """

    m = weighted_mean(x,e)

    v = np.sum((x-m)**2/e**2)/np.sum(1./e**2)
    return v

def norm(tod):
    """
    Normalise a TOD (Band,Sample)
    """
    rms = np.nanstd(tod,axis=1)
    mean = np.nanmean(tod,axis=1)

    return (tod-mean[:,None])/rms[:,None]


def downsample_time(tod,stepsize, binsize):
    """
    Downsample a TOD format: (Band,Sample) to a sampling time of binsize
    """

    Nsamples = int(binsize/stepsize)
    nbins = int(tod.shape[1]/Nsamples)

    binedges = np.arange(0,nbins+1)*binsize
    time = np.arange(tod.shape[1])*stepsize

    tod_out = np.zeros((tod.shape[0],nbins))

    for i in range(tod.shape[0]):
        tod_out[i] = np.histogram(time,binedges,weights=tod[i])[0]/np.histogram(time,binedges)[0]

    return time,(binedges[1:]+binedges[:-1])/2.,tod_out

def correlation(tod):
    """
    Create correlation matrix for 1 second steps in the data

    TOD : (Feed,Band,Sample)
    """
    tod = np.reshape(tod,(tod.shape[0]*tod.shape[1],tod.shape[-1]))
    time,mids,tod = downsample_time(tod,1./50., 1)
    tod = norm(tod)

    C = tod.dot(tod.T)/tod.shape[1]

    return C
