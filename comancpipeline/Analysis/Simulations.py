import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs
from scipy.optimize import fmin, leastsq
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust

from mpi4py import MPI 
comm = MPI.COMM_WORLD

import os
import healpy as hp
import scipy.fftpack as sfft

class SimulateSource(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,a=1., x0=0, y0=0, sx=4./60., sy=4./60., phi=0., offset=0, gx=0, gy=0, xoff=0, yoff=0 ):
        self.P = [a, xoff/60., sx, yoff/60., sy, phi*np.pi/180., offset, gx, gy]
        self.x0, self.y0 = x0, y0
    def __str__(self):
        return 'Simulating source at {:.2f} {:.2f}'.format(self.x0, self.y0)

    def run(self,data):

        self.simulate(data)

    def simulate(self,data):
        tod = data.getdset('spectrometer/tod')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        nHorns, nSBs, nChans, nSamples = tod.shape
        rot = hp.rotator.Rotator(rot=[self.x0, self.y0])
        for i in range(nHorns):
            decRot, raRot = rot((90-dec[i,:])*np.pi/180., ra[i,:]*np.pi/180.)
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] += Fitting.Gauss2dRotPlane(self.P, raRot*180./np.pi, (np.pi/2.-decRot)*180./np.pi,0,0)


class SimulateNoise(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,alpha = 1, tsys=40, fk=1.):
        self.tsys = tsys
        self.fk = fk
        self.alpha = alpha

    def __str__(self):
        return 'Simulating noise with Tsys: {:.1f}'.format(self.tsys) + r' f_knee: '+ '{:.2f}'.format(self.fk)

    def realisation(self, N):
        if not hasattr(self, 'f'):
            self.Nbig = 2**(int(np.log(N)/np.log(2)+1))
            self.f = sfft.fftfreq(self.Nbig, d=self.dT)
            self.f[0] = self.f[1]
        rms = self.tsys/np.sqrt(self.dT * self.dnu) * np.sqrt(2)

        wnoise = np.random.normal(size=self.Nbig, scale=rms)
        power  = (1. + (self.fk/np.abs(self.f))**self.alpha)
        noise  = sfft.ifft(sfft.fft(wnoise)*np.sqrt(power))

        return noise[:N] + self.tsys

    def run(self,data):

        self.simulate(data)

    def simulate(self,data):
        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9
        
        nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] = self.realisation(nSamples)
