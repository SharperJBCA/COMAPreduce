import numpy as np

c = 299792458.
k = 1.3806488e-23
h = 6.62606957e-34
T_cmb = 2.725
Jy = 1e26

def toJy(_nu,beam):
    '''
    toJy(nu,beam)

    nu: frequency in GHz
    beam: beam in steradians.
    
    '''

    nu =_nu* 1e9
    return 2.*k*nu**2/c**2 * beam * Jy

def planckcorr(nu_in):

    nu = nu_in * 1e9

    x = h*nu/k/T_cmb

    return x**2*np.exp(x)/(np.exp(x) - 1.)**2

def Units(unit,nu,pixbeam=1):
    """
    Converts units to K
    """
    if isinstance(nu,type(None)):
        return 1

    conversions = {'K':1.,
                   'mK_RJ':1e-3,
                   'mK':1e-3,
                   'mKCMB':planckcorr(nu)*1e-3,
                   'KCMB':planckcorr(nu),
                   'Wm2sr':1,
                   'MJysr':1e6/toJy(nu,1.)}

    if unit in conversions:
        return conversions[unit]
    else:
        return 1


def MAD(d,axis=0):
    """
    Return Median Absolute Deviation for array along one axis
    """
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48

    return rms

def auto_rms(tod : np.ndarray):
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
    rms =  auto_rms(tod) 
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
