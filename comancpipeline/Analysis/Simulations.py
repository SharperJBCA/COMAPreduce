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

from astropy import wcs as wcsModule
import h5py

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

class SimulateWhiteNoise(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, tsys=40, rfactor=1):
        '''
        rfactor - reduce noise by a given factor
        '''
        self.tsys = tsys
        self.rfactor = rfactor
        

    def __str__(self):
        return 'Simulating white noise with Tsys: {:.1f}'.format(self.tsys)

    def realisation(self, N):
        rms = self.tsys/np.sqrt(self.dT * self.dnu) * np.sqrt(2) * self.rfactor
        wnoise = np.random.normal(size=N, scale=rms)
        return wnoise + self.tsys

    def run(self,data):

        self.simulate(data)

    def simulate(self,data):
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')

        
        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9

        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            print('Creating new dataset')
            tod = np.zeros((nHorns, nSBs, nChans, nSamples))
        else:
            print('Reading existing dataset')
            tod = data.getdset('spectrometer/tod')

        for i in range(nHorns):
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] += self.realisation(nSamples)

        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            data.updatedset('spectrometer/tod', tod)
            data.setAttr('comap', 'sim_tod', True)

class SimulateSignal(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,cocubefilename=None, ra0=0, dec0=0):
        self.cocubefilename = cocubefilename
        self.ra0 = ra0
        self.dec0=dec0

    def __str__(self):
        return 'Simulating signal from {}'.format(self.cocubefilename)

    def run(self,data):
        self.simulate(data)

    def readCOCube(self):
        '''
        '''
        self.cubedataFile = h5py.File(self.cocubefilename,'r')
        self.wcs = wcsModule.WCS(naxis=2) 
        self.wcs.wcs.crval = [self.ra0, self.dec0]
        self.wcs.wcs.cdelt = self.cubedataFile['cube'].attrs['cdelt']
        self.wcs.wcs.crpix = self.cubedataFile['cube'].attrs['crpix']
        self.wcs.wcs.ctype = [self.cubedataFile['cube'].attrs['ctype1'],
                              self.cubedataFile['cube'].attrs['ctype2'] ]
        self.nxpix = self.cubedataFile['cube'].attrs['nxpix']
        self.nypix = self.cubedataFile['cube'].attrs['nypix']

        self.cocube = self.cubedataFile['cube'][...]

        self.coFreq = self.cubedataFile['frequencies'][:]
        return self.cubedataFile, self.wcs

    def getPixelCoords(self,ra, dec):
        '''
        '''
        ypixels, xpixels = self.wcs.wcs_world2pix(ra,dec,0)
        pflat  = (xpixels.astype(int) + self.nxpix*ypixels.astype(int)).astype(int)
        gd = (xpixels >= 0) & (xpixels < self.nxpix) & (ypixels >= 0) & (ypixels < self.nypix)
        pflat[~gd] = -1
        return pflat, np.where(~(gd))[0]

    def cube2TOD(self, pixels, channel, masked):
        '''
        '''
        tod = self.cocube[pixels, channel]
        tod[masked] = 0
        return tod

    def simulate(self,data):
        '''
        '''
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')

        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            print('Creating new dataset')
            tod = np.zeros((nHorns, nSBs, nChans, nSamples))
        else:
            print('Reading existing dataset')
            tod = data.getdset('spectrometer/tod')

        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9

        cocube, wcs = self.readCOCube()

        # map indices of COCube frequencies to those in real data.
        self.nuMap = [[np.argmin((nui-self.coFreq)**2) for nui in sb] for sb in nu]
        
        #nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            pflat, masked = self.getPixelCoords(ra[i,:], dec[i,:])
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] += self.cube2TOD(pflat, self.nuMap[j][k], masked)
                    
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            data.updatedset('spectrometer/tod', tod)
            data.setAttr('comap', 'sim_tod', True)

        self.cubedataFile.close()
