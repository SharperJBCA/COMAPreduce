import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools import  binFuncs, stats

class Offsets:
    """
    Stores offset information
    """
    def __init__(self,offset, Noffsets, Nsamples):
        """
        """
        
        self.Noffsets= int(Noffsets)
        self.offset_length = int(offset)
        self.Nsamples = int(Nsamples )

        self.values = np.zeros(self.Noffsets)

        self.sig = np.zeros(self.Noffsets)
        self.wei = np.zeros(self.Noffsets)

        self.offsetpixels = np.arange(self.Nsamples)//self.offset_length

    def __getitem__(self,i):
        """
        """
        return self.values[i//self.offset_length]

    def __call__(self):
        return np.repeat(self.values, self.offset_length)[:self.Nsamples]


    def clear(self):
        self.values *= 0
        self.sig *= 0
        self.wei *= 0

    def accumulate(self,tod,weights,chunk):
        """
        Add more data to residual offset
        """
        binFuncs.binValues(self.sig, self.offsetpixels[chunk[0]:chunk[1]], weights=tod*weights )
        binFuncs.binValues(self.wei, self.offsetpixels[chunk[0]:chunk[1]], weights=weights    )


    def average(self):
        self.good = np.where((self.wei != 0 ))[0]
        self.values[self.good] = self.sig[self.good]/self.wei[self.good]
