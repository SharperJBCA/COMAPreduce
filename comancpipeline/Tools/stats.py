import numpy as np
from matplotlib import pyplot
import h5py
import healpy as hp
import sys
from tqdm import tqdm
from comancpipeline.Tools import Coordinates
from matplotlib.transforms import ScaledTranslation
from scipy.signal import fftconvolve
import seaborn as sns


def MAD(d,axis=0):
    """
    Return Median Absolute Deviation for array along one axis
    """
    med_d = np.nanmedian(d,axis=axis)
    rms = np.sqrt(np.nanmedian((d-med_d)**2,axis=axis))*1.48

    return rms
