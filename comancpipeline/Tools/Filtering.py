# Function for filtering either 1D or 2D data.
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal

def estimateBackground(_tod, rms, close=None, sampleRate=50, cutoff=1.):
    """
    Takes the TOD and set of indices describing the location of the source.
    Fits polynomials beneath the source and then applies a low-pass filter to the full
    data. It returns this low pass filtered data
    """
    time = np.arange(_tod.size)
    tod = _tod*1.
    
    # First we will find the beginning and end samples the source crossings
    if isinstance(close, type(None)):
        close = np.zeros(_tod.size).astype(bool)
        close[:sampleRate] = True
        close[-sampleRate:] = True
    timeFit = time[close]
    timeZones = (timeFit[1:] - timeFit[:-1])
    timeSelect= np.where((timeZones > 5))[0]

    closeIndex = np.where(close)[0]
    indices = (closeIndex[:-1])[timeSelect]
    indices = np.concatenate((closeIndex[0:1], indices, (np.where(close)[0][:-1])[timeSelect+1], [closeIndex[-1]]))
    indices = np.sort(indices)
                
    # For each source crossing fit a polynomial using the data just before and after
    
    for m in range(indices.size//2):
        lo, hi = indices[2*m], indices[2*m+1]
        lo = max([lo, 0])
        hi = min([hi, tod.size])
        fitRange = np.concatenate((np.arange(np.max([lo-sampleRate,0]),lo), np.arange(hi, np.min([hi+sampleRate, tod.size]))  )).astype(int)
        #fitRange = np.concatenate((np.arange(lo-sampleRate,lo), np.arange(hi, hi+sampleRate)  )).astype(int)

        dmdl = np.poly1d(np.polyfit(time[fitRange], tod[fitRange],3))
        tod[lo:hi] = np.random.normal(scale=rms, loc=dmdl(time[lo:hi]))
          
    # apply the low-pass filter
    Wn = cutoff/(sampleRate/2.)
    b, a = signal.butter(4, Wn, 'low')
    background = signal.filtfilt(b, a, tod[:])


    return background

def estimateAtmosphere(_tod, el, rms, close=None, sampleRate=50, cutoff=1.):
    """
    Takes the TOD and set of indices describing the location of the source.
    Fits polynomials beneath the source and then applies a low-pass filter to the full
    data. It returns this low pass filtered data
    """
    time = np.arange(_tod.size)
    tod = _tod*1.
    
    # First we will find the beginning and end samples the source crossings
    if isinstance(close, type(None)):
        close = np.zeros(_tod.size).astype(bool)
        close[:sampleRate] = True
        close[-sampleRate:] = True
    timeFit = time[close]
    timeZones = (timeFit[1:] - timeFit[:-1])
    timeSelect= np.where((timeZones > 5))[0]

    closeIndex = np.where(close)[0]
    indices = (closeIndex[:-1])[timeSelect]
    indices = np.concatenate((closeIndex[0:1], indices, (np.where(close)[0][:-1])[timeSelect+1], [closeIndex[-1]]))
    indices = np.sort(indices)
                
    # For each source crossing fit a polynomial using the data just before and after
    
    for m in range(indices.size//2):
        lo, hi = indices[2*m], indices[2*m+1]
        lo = max([lo, 0])
        hi = min([hi, tod.size])
        fitRange = np.concatenate((np.arange(lo-sampleRate,lo), np.arange(hi, hi+sampleRate))).astype(int)
        dmdl = np.poly1d(np.polyfit(time[fitRange], tod[fitRange],3))
        tod[lo:hi] = np.random.normal(scale=rms, loc=dmdl(time[lo:hi]))
    # apply the low-pass filter
    A = 1./np.sin(el*np.pi/180.)

    dmdl = np.poly1d(np.polyfit(A, tod,1))

    return dmdl(A)



def removeNaN(d):
    """
    Fills NaN values with neighbouring values

    args : d (arraylike)
    """
    dnan = np.where(np.isnan(d))[0]
    dgd  = np.where(~np.isnan(d))[0]
    if len(dnan) > 0:
        for nanid in dnan:
            d[nanid] = (d[dgd])[np.argmin((dgd-nanid))]
            d[dnan] = d[dnan-1]


def calcRMS(tod):
    """
    Estimate rms of TOD using adjacent pairs for the last array dimension

    args: tod (arraylike, can be multidimensional)
    """
    nSamps = tod.shape[-1]
    # Calculate RMS from adjacent pairs
    splitTOD = (tod[...,:(nSamps//2) * 2:2] - tod[...,1:(nSamps//2)*2:2])
    rms = np.nanstd(splitTOD,axis=-1)/np.sqrt(2)
    return rms


def noiseProperties(tod, ra, dec, mjd):
    """
    Calculates rms of TOD using adjacent pairs for the last array dimension, and corresponding RA and Dec for use in noise map
    """

    nSamps = tod.shape[-1]
    rms = (tod[...,:(nSamps//2) * 2:2] - tod[...,1:(nSamps//2)*2:2])
    ranew = (ra[...,1:(nSamps//2)*2:2] + ra[...,:(nSamps//2)*2:2]) / 2
    decnew = (dec[...,1:(nSamps//2)*2:2] + dec[...,:(nSamps//2)*2:2]) / 2
    mjdnew = (mjd[...,1:(nSamps//2)*2:2] + mjd[...,:(nSamps//2)*2:2]) / 2
    return rms, ranew, decnew, mjdnew
