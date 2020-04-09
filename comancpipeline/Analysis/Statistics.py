import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types
from os import listdir, getcwd
from os.path import isfile, join
from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import pandas as pd
#from mpi4py import MPI 
import os
#comm = MPI.COMM_WORLD
from scipy.optimize import minimize

from tqdm import tqdm

class FnoiseStats(DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, nbins=50, samplerate=50):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        self.nbins = 50
        self.samplerate=50

    def run(self,data):
        """
        Expects a level2 file structure to be passed.
        """
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod = data['level2/averaged_tod'][...]
        features = data['level1/spectrometer/features'][:]
        uf, counts = np.unique(features,return_counts=True) # select most common feature
        ifeature = np.floor(np.log10(uf[np.argmax(counts)])/np.log10(2))
        selectFeature = self.featureBits(features.astype(float), ifeature)
        el = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]

        features[features == 0] = 0.1
        # snip to features
        tod = tod[...,selectFeature]
        el  = el[...,selectFeature]

        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape

        self.powerspectra = np.zeros((nFeeds, nBands, nChannels, self.nbins))
        self.freqspectra = np.zeros((nFeeds, nBands, nChannels, self.nbins))
        self.fnoise_fits = np.zeros((nFeeds, nBands, nChannels, 2))
        self.wnoise_auto = np.zeros((nFeeds, nBands, nChannels, 1))
        self.atmos = np.zeros((nFeeds, nBands, nChannels, 2))

        pbar = tqdm(total=((nFeeds-1)*nBands*nChannels))
        for ifeed in range(nFeeds):
            if feeds[ifeed] == 20:
                continue
            for iband in range(nBands):
                for ichan in range(nChannels):
                    if np.nansum(tod[ifeed, iband, ichan,:]) == 0:
                        continue
                    tod[ifeed,iband,ichan], atmos = self.RemoveAtmosphere(tod[ifeed,iband,ichan], 
                                                                          el[ifeed])

                    ps, nu, f_fits, w_auto = self.FitPowerSpectrum(tod[ifeed,iband,ichan,:])
                    self.powerspectra[ifeed,iband,ichan,:] = ps
                    self.freqspectra[ifeed,iband,ichan,:]  = nu
                    self.fnoise_fits[ifeed,iband,ichan,:]  = f_fits
                    self.wnoise_auto[ifeed,iband,ichan,:]  = w_auto
                    self.atmos[ifeed,iband,ichan,:] = atmos[0],atmos[1]
                    pbar.update(1)
        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        allowed_sources = ['fg{}'.format(i) for i in range(10)] + ['GField{:02d}'.format(i) for i in range(20)]
        source = data['level1/comap'].attrs['source'].decode('utf-8')
        comment = data['level1/comap'].attrs['comment'].decode('utf-8')
        print('SOURCE', source)
        if not source in allowed_sources:
            return data
        if 'Sky nod' in comment:
            return data

        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        self.run(data)
        self.write(data)

        return data


    def AutoRMS(self, tod):
        """
        Calculate auto-pair subtracted RMS of tod
        """
        #N2 = tod.size//2*2
        #diff = tod[1:N2:2]-tod[0:N2:2]
        N4 = tod.size//4*4
        ABBA = tod[0:N4:4] - tod[1:N4:4] - tod[2:N4:4] + tod[3:N4:4]
        med = np.nanmedian(ABBA)
        mad = np.sqrt(np.nanmedian(np.abs(ABBA-med)**2))*1.4826/np.sqrt(4)
        return mad
        
    def PowerSpectrum(self, tod):
        """
        Calculates the bin averaged power spectrum
        """
        nu = np.fft.fftfreq(tod.size, d=1/self.samplerate)
        binEdges = np.logspace(np.log10(nu[1]), np.log10(nu[nu.size//2-1]), self.nbins+1)
        ps     = np.abs(np.fft.fft(tod))**2/tod.size
        counts = np.histogram(nu[1:nu.size//2], binEdges)[0]
        signal = np.histogram(nu[1:nu.size//2], binEdges, weights=ps[1:nu.size//2])[0]
        freqs  = np.histogram(nu[1:nu.size//2], binEdges, weights=nu[1:nu.size//2])[0]

        return freqs/counts, signal/counts, counts

    def Model(self, P, x, rms):
        return rms**2 * (1 + (x/10**P[0])**P[1])
        
    def Error(self, P, x, y,e, rms):
        error = np.abs(y/e)
        chi = (np.log(y) - np.log(self.Model(P,x,rms)))/error
        return np.sum(chi**2)

    def FitPowerSpectrum(self, tod):
        """
        Calculate the power spectrum of the data, fits a 1/f noise curve, returns parameters
        """
        auto_rms = self.AutoRMS(tod)
        nu, ps, counts = self.PowerSpectrum(tod)

        # Only select non-nan values
        # You may want to increase min counts, 
        # as the power spectrum is non-gaussian for small counts
        good = (counts > 50) & ( (nu < 0.03) | (nu > 0.05))

        args = (nu[good], ps[good],auto_rms/np.sqrt(counts[good]), auto_rms)
        bounds =  [[None,None],[-3,0]]
        P0 = [0,-1]
        P1 = minimize(self.Error, P0, args= args, bounds = bounds)


        return ps, nu, P1.x, auto_rms

    def RemoveAtmosphere(self, tod, el):
        """
        Remove 1/sin(E) relationship from TOD
        """
        A = 1/np.sin(el*np.pi/180) # Airmass
        pmdl = np.poly1d(np.polyfit(A, tod,1))
        return tod- pmdl(A), pmdl

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """        

        if not 'level2' in data:
            return 

        lvl2 = data['level2']
        dnames = ['fnoise_fits','wnoise_auto', 'powerspectra','freqspectra', 'atmos']
        dsets = [self.fnoise_fits,self.wnoise_auto,self.powerspectra,self.freqspectra,self.atmos]
        for (dname, dset) in zip(dnames, dsets):
            if dname in lvl2:
                del lvl2[dname]            
            lvl2.create_dataset(dname,  data=dset)
