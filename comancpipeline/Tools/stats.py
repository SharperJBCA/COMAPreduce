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
