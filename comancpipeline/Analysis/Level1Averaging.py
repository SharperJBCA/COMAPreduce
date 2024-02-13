#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:17:03 2023

AtmosphereRemoval - 
Level1Averaging -
Level1AveragingGainCorrection - 
%% TODO 
- [ ] The gain correction function 'fit_gain_fluctuations' currently requires
all four bands to be present. Does it work if it is performed per band? Otherwise
we lose all four bands if any one roach is offline. 

@author: sharper
"""


from matplotlib import pyplot
import os 

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 
from .PowerSpectra import FitPowerSpectrum
from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2, COMAPLevel1
from scipy.sparse import linalg, block_diag
from scipy.ndimage import gaussian_filter1d

from comancpipeline.Tools.median_filter import medfilt
from scipy.optimize import minimize
import logging 
import warnings
from comancpipeline.Tools import Coordinates 
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from comancpipeline.Analysis.VaneCalibration import MeasureSystemTemperature
from comancpipeline.Analysis import GainSubtraction 





@dataclass 
class SkyDip(PipelineFunction):
    name : str = 'SkyDip'
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2())
    groups : list = field(default_factory=lambda: ['skydip'])
    overwrite : bool = False
    STATE : bool = True
    figure_directory : str = 'figures'

    fit_values : dict = field(default_factory=lambda: None)
    @property
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {'skydip/fit_values':self.fit_values}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
        if isinstance(data, COMAPLevel2):
            return self.STATE 

        if not data.source_name in Coordinates.CalibratorList: 
            self.skydip(data)

        return self.STATE

    def skydip(self, data):

        # First we have to check if the previous observation is a skydip
        this_obsid = data.obsid
        previous_obsid = this_obsid - 1

        # Search for the previous observation
        previous_data = COMAPLevel1(overwrite=False, large_datasets=['spectrometer/tod']) 
        data_dir = os.path.dirname(data.filename)
        if not previous_data.read_data_file_by_obsid(previous_obsid, data_dir): # If the file doesn't exist, return
            return 

        if 'sky nod' in previous_data.comment.lower():
            self.fit_skydip(data, previous_data,figure_path=self.figure_directory)
        
    def fit_skydip(self, data, previous_data, figure_path='figures/'):
        figure_directory = f'{figure_path}/{data.obsid}/'
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        n_feeds, n_bands, n_channels, n_tod = previous_data.tod_shape 
        
        features = previous_data.features
        scan = np.where((features == 8))[0]
        A = previous_data.airmass[:,scan]
        el = previous_data.el[:,scan]
        if len(scan) == 0:
            self.STATE = False
            return

        tsys = self.level2.system_temperature
        gain = self.level2.system_gain
        if gain.shape[0] == 0:
            self.STATE = False
            return 
        tsys_el = self.level2.system_temperature_el
        #print(previous_data.system_temperature)
        self.fit_values = np.zeros((n_feeds, n_bands, n_channels, 2))
        for ((ifeed, feed),) in tqdm(previous_data.tod_loop(bands=False, channels=False), desc='Sky Dip Fit Loop'):
            if (feed > 19):
                continue
            _tod = previous_data['spectrometer/tod'][ifeed,...]
            tod = _tod[...,scan]
            try:
                tod /= gain[0,ifeed,...,None]
            except IndexError:
                print('##########################################')
                print('Gain not found', data.filename, data.obsid)
                print('##########################################')
                raise IndexError('Gain not found')
            select = np.where((el[ifeed] > 40) & (el[ifeed] < 55))[0]
            print('MIN EL', np.min(el[ifeed]), 'MAX EL', np.max(el[ifeed])) 
            logging.debug(f'{self.name}: MIN EL/MAX EL FOR FEED {feed:02d} is {np.min(el[ifeed]):.2f}/{np.max(el[ifeed]):.2f}')

            for iband in range(n_bands):
                for ichannel in range(n_channels):
                    try:
                        fits = np.polyfit(A[ifeed,select], tod[iband,ichannel,select], 1)
                        self.fit_values[ifeed, iband, ichannel] = fits
                    except (np.linalg.LinAlgError, TypeError):
                        self.fit_values[ifeed, iband, ichannel] = np.nan

            fig, ax = pyplot.subplots(2,1, sharex=True)

            nu = previous_data.frequency.flatten()
            idx = np.argsort(nu)
            ax[0].plot(nu[idx], self.fit_values[ifeed,...,1].flatten()[idx])
            ax[1].plot(nu[idx], self.fit_values[ifeed,...,0].flatten()[idx])
            ax[1].set_xlabel('Frequency [GHz]')
            ax[1].set_ylabel('Sky Brightness [K] ')
            ax[0].set_ylabel('Zero System Temperature [K]')
            ax[0].set_xlim(26, 34)
            ax[1].set_xlim(26, 34)

            ax[0].set_ylim(0,100)
            ax[1].set_ylim(0,30)

            pyplot.tight_layout()
            pyplot.suptitle(f'Feed {feed:02d}')
            pyplot.savefig(f'{figure_directory}/skydip_feed{feed:02d}.png')
            pyplot.close(fig)
@dataclass 
class AtmosphereRemoval(PipelineFunction):    
    
    name : str = 'AtmosphereRemoval'
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    groups : list = field(default_factory=lambda: ['atmosphere']) 

    figure_directory : str = 'figures' 
    overwrite : bool = False 
    STATE : bool = True 



    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {'atmosphere/fit_values':self.fit_values}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 

        if isinstance(data, COMAPLevel2):
            self.fit_values = data['atmosphere/fit_values'][...]
            return self.STATE 

        self.filter_atmosphere(data)
        
        return self.STATE
    
    @staticmethod 
    def subtract_fitted_atmosphere(A, tod, fit_values):
        """Returns the atmosphere subtracted data"""
        Nfreq = tod.shape[0]
        Z = np.ones((A.size, 2))
        Z[:,1] = A
        Z = block_diag([Z]*Nfreq)
        return tod -  np.reshape(Z.dot(fit_values.T.flatten()[:,None]),tod.shape)

    def fit_atmosphere(self, A, tod, lowf=10,highf=1024-10,MINIMUM_CHUNK_SIZE=100):
        """Fit for the atmosphere"""

        # Nfreq = highf - lowf
        select = np.arange(lowf,highf,dtype=int) 
        select = np.delete(select, np.arange(select.size//2-2,select.size//2+3,dtype=int))
        Nfreq = select.size
        select_time = np.isfinite(np.sum(tod[select],axis=0))
        Acut = A[select_time]
        
        if Acut.size < MINIMUM_CHUNK_SIZE:
            return np.zeros(1024) + np.nan, np.zeros(1024) + np.nan
        Zmasked = np.ones((Acut.size, 2))
        Zmasked[:,1] = Acut
        Zmasked = block_diag([Zmasked]*Nfreq)
        d = tod[select] 
        d = d[:,select_time].flatten()[:,None]

        try:
            b = Zmasked.T.dot(d)
        except ValueError:
            logging.INFO(f'{self.name}: Value Error in Atmosphere Fit')
            return np.zeros(Nfreq), np.zeros(Nfreq)
        M = (Zmasked.T.dot(Zmasked))
        fit_values = linalg.spsolve(M,b)
        offset = np.zeros(1024) + np.nan
        atmos = np.zeros(1024) + np.nan
        offset[select] = fit_values[::2] 
        atmos[select] = fit_values[1::2]

        return offset, atmos
    
    def filter_atmosphere(self, data : HDF5Data):
        """ """
        A = data.airmass
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape 
        n_scans = len(data.scan_edges)
        self.fit_values = np.zeros((n_scans, n_feeds, n_bands, 2, n_channels))
        logging.info(f'{self.name}: Total number of scans {n_scans:03d}')
        
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='Atmosphere Filter Feed Loop'):
            _tod = data['spectrometer/tod'][ifeed,...]
            for iband in range(n_bands):
                tod = _tod[iband]
                for iscan, (start, end) in enumerate(data.scan_edges):
                    if all(data.features[start:end] == 9): # ignore constant elevation scans
                        self.fit_values[iscan,ifeed,iband,0] = np.nanmedian(tod[...,start:end],axis=-1)
                        continue 
                    self.fit_values[iscan,ifeed,iband] = self.fit_atmosphere(A[ifeed,start:end], tod[...,start:end])
                    logging.debug(f'{self.name}: MEAN ATMOS FIT FOR FEED {feed:02d} in SCAN {iscan:02d} is {np.nanmean(self.fit_values[iscan,ifeed,iband,1]):.1f}')


@dataclass 
class Level1Averaging(PipelineFunction):    
    name : str = 'Level1Averaging'

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 
    
    tod : np.ndarray = field(default_factory=lambda : np.zeros(1))
    tod_stddev : np.ndarray = field(default_factory=lambda : np.zeros(1))

    frequency_mask : np.ndarray = field(default_factory=lambda : np.zeros(1,dtype=bool))

    frequency_bin_size : int = 512 
    
    N_CHANNELS : int = 1024
    STATE : bool = True 

    def __post_init__(self):
        
        self.frequency_mask = np.zeros(self.N_CHANNELS,dtype=bool)
        self.frequency_mask[:10] = True
        self.frequency_mask[-10:]= True 
        self.frequency_mask[511:514] = True
        
        self.channel_edges = np.arange(self.N_CHANNELS//self.frequency_bin_size + 1)
        self.channel_idx = np.arange(self.N_CHANNELS)

    def __call__(self, data : HDF5Data) -> HDF5Data:
        if isinstance(data, COMAPLevel2):
            return self.STATE 
                
        self.average_tod(data)
        
        return self.STATE
    
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {'spectrometer/tod': self.tod,
                'spectrometer/tod_stddev': self.tod_stddev}
        attrs = {}

        return data, attrs

    def average_tod(self, data : HDF5Data) -> HDF5Data:
        """Average the time ordered data in frequency"""
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        n_channels_low = n_channels//self.frequency_bin_size 

        self.tod = np.zeros((n_feeds, n_bands, n_channels_low, n_tod))
        self.tod_stddev = np.zeros((n_feeds, n_bands, n_channels_low, n_tod))

        for (ifeed, feed), iband in tqdm(data.tod_loop(channels=False), desc='TOD Averaging Loop'):
            tod = data['spectrometer/tod'][ifeed, iband, ...]
            tod /= (self.level2.system_gain)[0,ifeed,iband,:,None]
            weights = (1./self.level2.system_temperature**2)[0,ifeed,iband,:,None]
            
            # Mask edge channels
            weights[self.frequency_mask,:] = 0
            
            tod_avg = np.reshape(tod*weights,  ( n_channels_low, self.frequency_bin_size, n_tod))
            tod_sqr_avg = np.reshape(tod**2*weights, ( n_channels_low, self.frequency_bin_size, n_tod))
            tod_wei = np.reshape(weights,  ( n_channels_low, self.frequency_bin_size))
            
            tod_avg = np.sum(tod_avg,axis=1)/np.sum(tod_wei,axis=1)[:,None]
            tod_sqr_avg = np.sum(tod_sqr_avg,axis=1)/np.sum(tod_wei,axis=1)[:,None]
                    
            stddev = np.sqrt(tod_sqr_avg - tod_avg**2)
            
            self.tod[ifeed, iband, ...] = tod_avg
            self.tod_stddev[ifeed, iband, ...] = stddev 
            
@dataclass 
class CheckLevel1File(PipelineFunction):
    name : str = 'CheckLevel1File'
    groups : list = field(default_factory=lambda: []) 
    
    overwrite : bool = True
    STATE : bool = True 
    MIN_TIME : float = 300 # s
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {}
        attrs = {}

        return data, attrs
    
    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
        
        comment = data.attrs('comap','comment')
        if 'sky dip' in comment.lower(): 
            logging.info(f'Observation is a sky dip. (comment: {comment})')
            self.STATE = False 
        if 'sky nod' in comment.lower(): 
            logging.info(f'Observation is a sky dip. (comment: {comment})')
            self.STATE = False 

        # Check the file is longer than 5 minutes
        mjd0 = data['spectrometer/MJD'][0] 
        mjd1 = data['spectrometer/MJD'][-1] 
        time = (mjd1-mjd0)*24*3600. 
        if time < self.MIN_TIME: 
            logging.info(f'File contains only {time:.0f} seconds of data (< {self.MIN_TIME}s).')
            self.STATE = False 
        return self.STATE




@dataclass 
class Level1AveragingGainCorrectionOctober2023(Level1Averaging):
    name : str = 'Level1AveragingGainCorrection'
    groups : list = field(default_factory=lambda: ['averaged_tod']) 
    figure_directory : str = 'figures'

    overwrite : bool = False
    STATE : bool = True 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {'averaged_tod/tod':self.tod_cleaned,
                 'averaged_tod/tod_original':self.tod_original,
                 'averaged_tod/weights':self.tod_weights,
                 'averaged_tod/scan_edges':self.scan_edges,
                 'averaged_tod/frequency_power_spectra':self.freq_power_spectra,
                 'averaged_tod/frequency_power_spectra_fits':self.freq_power_spectra_fits}
                 
        attrs = {}

        return data, attrs
    
    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
        self.average_tod(data, level2_data)
        
        return self.STATE
    
    def build_filters(self,start, end):
        # Fix NaNs, assign NaNs to nearest next good value along the time axis
        data_reshape = tod[...,start:end].reshape((n_bands*n_channels, end-start))
        nan_tod = np.isnan(data_reshape) 
        ones = np.ones(data_reshape.shape)*np.nanmedian(data_reshape,axis=1)[:,None] 
        data_reshape[nan_tod] = ones[nan_tod]

        tod[...,start:end] = data_reshape.reshape((n_bands, n_channels, end-start))
        print('ORIGINAL TOD', np.nanstd(tod))

        if data.source_name in Coordinates.CalibratorList: 
            clean_tod = tod[...,start:end] - np.nanmedian(tod[...,start:end],axis=-1)[...,None]
        else:
            clean_tod = self.remove_atmosphere(data.airmass[ifeed,start:end], tod[...,start:end], level2_data['atmosphere/fit_values'][iscan,ifeed,:])
        print('AFTER AIR', np.nanstd(clean_tod))
        clean_tod, normalisation_factor = self.normalise_data(clean_tod, level2_data['vane/system_temperature'][0,ifeed,:,:], level2_data['vane/system_gain'][0,ifeed,:,:]) 
        print('AFTER NORMALISATION', np.nanstd(clean_tod))
        clean_tod = self.median_filter(clean_tod, int(50*120))
        print('AFTER MEDIAN FILTER', np.nanstd(clean_tod))
        #frequency_power_spectra = self.frequency_spectra_per_band(clean_tod)
        try:
            dG = self.gain_subtraction(data, clean_tod, level2_data['vane/system_temperature'][0,ifeed,:,:]) 
        except (ValueError, IndexError): 
            dG = None

        # Build weights, mask out edge channels and centre channels
        weights = 1./level2_data.system_temperature[0,ifeed]**2 
        weights[level2_data.system_temperature[0,ifeed] == 0] = 0
        weights[:,:10] = 0
        weights[:,-10:] = 0
        weights[:,510:515] = 0
        if not isinstance(dG, type(None)):
            print('AFTER GAIN', np.nanstd(dG))

        if not isinstance(dG, type(None)):
            residual = (clean_tod - dG[None,None,:])*normalisation_factor/level2_data['vane/system_gain'][0,ifeed,:,:,None]
        else:
            residual = clean_tod*normalisation_factor/level2_data['vane/system_gain'][0,ifeed,:,:,None]

        residual = self.weighted_average_over_band(residual, weights) 

        if (iscan == 0) & (not isinstance(dG, type(None))):
            avg_tod = self.weighted_average_over_band(clean_tod, weights) 
            ### Plot some diagnostics
            self.plot_gain_examples(data, clean_tod[0], avg_tod[0], dG, level2_data, iscan, feed,
                    level2_data.system_temperature[0,ifeed,0,:,None],figure_path=self.figure_directory)
            ### 
        clean_tod = clean_tod*level2_data.system_temperature[0,ifeed,:,:,None]
        avg_tod = self.weighted_average_over_band(clean_tod, weights) 

        self.tod_cleaned[ifeed,:,start:end]  = residual
        self.tod_original[ifeed,:,start:end] = avg_tod
        self.tod_weights[ifeed,:,start:end]  = 1./self.auto_rms(residual)[:,None]**2


    def average_tod(self, data : HDF5Data, level2_data : COMAPLevel2) -> HDF5Data:
        """Average the time ordered data in frequency"""
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        n_channels_low = 4 # n_channels//self.frequency_bin_size 

        bandwidth = (2e9/n_channels) 
        sample_rate = 50. 
        self.tod_cleaned  = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_original = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_weights  = np.zeros((n_feeds, n_channels_low, n_tod))
        self.scan_edges   = data.scan_edges
        n_scans = len(self.scan_edges) 
        self.freq_power_spectra = np.zeros((n_scans, n_feeds, n_bands, 15,2))
        self.freq_power_spectra_fits = np.zeros((n_scans, n_feeds, n_bands, 3))
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='TOD Averaging Loop'):
            if (feed > 19):
                continue
            tod = data['spectrometer/tod'][ifeed, ...]

            for iscan, (start, end) in enumerate(self.scan_edges): 
                logging.debug(f'{self.name}: Averaging level 1 {iscan:02d} in Feed {feed:02d}')
                tod[...,start:end] = self.fill_bad_data(tod[...,start:end])



@dataclass 
class Level1AveragingGainCorrection(Level1Averaging):
    name : str = 'Level1AveragingGainCorrection'
    groups : list = field(default_factory=lambda: ['averaged_tod']) 
    figure_directory : str = 'figures'

    overwrite : bool = False
    STATE : bool = True 

    gain_subtraction_name : str = 'gain_subtraction_fit' # Default gain subtraction name 
    gain_subtracted_tod_name : str = 'tod'

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        self.data  = {f'averaged_tod/{self.gain_subtracted_tod_name}':self.tod_cleaned,
                 'averaged_tod/tod_original':self.tod_original,
                 'averaged_tod/weights':self.tod_weights,
                 'averaged_tod/scan_edges':self.scan_edges,
                 'averaged_tod/frequency_power_spectra':self.freq_power_spectra,
                 'averaged_tod/frequency_power_spectra_fits':self.freq_power_spectra_fits}
                 
        self.attrs = {}

        return self.data, self.attrs
    
    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
        if isinstance(data, COMAPLevel2):
            self.tod_cleaned = data[f'averaged_tod/{self.gain_subtracted_tod_name}'],
            self.tod_original = data['averaged_tod/tod_original'],
            self.tod_weights = data['averaged_tod/weights'],
            self.scan_edges = data['averaged_tod/scan_edges'],
            self.freq_power_spectra = data['averaged_tod/frequency_power_spectra'],
            self.freq_power_spectra_fits = data['averaged_tod/frequency_power_spectra_fits']
            return self.STATE
        self.average_tod(data, level2_data)
        
        return self.STATE
    
    def auto_rms(self, tod : np.ndarray):
        """ Calculate rms from differences of adjacent samples """ 
        
        N = tod.shape[-1]//2 * 2
        diff = tod[...,1:N:2] - tod[...,0:N:2]
        
        return np.nanstd(diff,axis=-1)/np.sqrt(2) 
        
    def normalised_data(self, data : HDF5Data, tod : np.ndarray):
        """ """ 
        if (tod.shape[-1] == 0):
            return None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tod_mean = np.nanmean(tod, axis=2)
            tod_std  = self.auto_rms(tod) 
        
        tod_norm = (tod - tod_mean[...,None])/tod_std[...,None]
        
        return tod_norm

    def bin_power_spectrum(self, ps_nu, ps, nbins=15):
        # Bin the power spectrum
        nu_edges = np.logspace(np.log10(np.min(ps_nu[1:ps.size//2])),np.log10(np.max(ps_nu)),nbins+1)
        top = np.histogram(ps_nu,nu_edges,weights=ps)[0]
        bot = np.histogram(ps_nu,nu_edges)[0]
        gd = (bot != 0)
        P_bin = np.zeros(bot.size) + np.nan
        nu_bin = np.zeros(bot.size) + np.nan
        nu_bin[gd] = np.histogram(ps_nu,nu_edges,weights=ps_nu)[0][gd]/bot[gd]
        P_bin[gd] = top[gd]/bot[gd]
        
        # Compute the power law fit
        gd = (bot != 0) & np.isfinite(P_bin) & (nu_bin != 0)
        nu_bin = nu_bin[gd]
        P_bin = P_bin[gd]

        return nu_bin, P_bin
    
    def fit_power_spectrum(self, data : HDF5Data, tod : np.ndarray , lowf=10, highf=-10):
        """ Fits the averaged powerspectrum """
        def model(P,x):
            """
            Assuming model \sigma_w^2 + \sigma_r^2 (frequency/frequency_r)^\alpha
            Parameters
            ----------
            Returns
            -------
            
            """
            return P[0] + (1+np.abs(x/P[1])**P[2])

        def error(P,x,y,sig2):
        
            chi2 = np.sum((np.log(y)-np.log(model([sig2,P[0],P[1]],x)))**2)
            if not np.isfinite(chi2):
                print(P) 
            
            return chi2
        # Calculate the average of the TOD
        resid_avg_orig = np.nanmean(tod[10:-10],axis=0)
        
        # Calculate the power spectrum
        ps = np.abs(np.fft.fft(resid_avg_orig)**2)
        ps_nu = np.fft.fftfreq(ps.size,d=1./50.)
        
        nu_bin, P_bin = self.bin_power_spectrum(ps_nu, ps, nbins=15)
        
        if len(nu_bin) == 0:
            raise IndexError
        
        P0 = [0.1, -1]
        gd = (nu_bin > 0.1) # just the high frequencies
        result = minimize(error,P0,args=(nu_bin[gd],P_bin[gd],P_bin[-1]),bounds=([0,None],[None,0]))
        
        results = [P_bin[-1], result.x[0],result.x[1]]
        return results
        
    
    def weighted_average_over_band(self, residual, weights):
        weights[:,:50] = 0 
        weights[:,-50:]= 0
        weights[:,512] = 0
        weights[np.isnan(residual[...,0])] =0 
        residual[np.isnan(residual)] =0 
        residual = np.sum(residual*weights[...,None],axis=1)/np.sum(weights[...,None],axis=1) 
        return residual

    @staticmethod 
    def median_filter(tod,medfilt_stepsize):
        """
        Calculate this AFTER removing the atmosphere.
        """
        if any(~np.isfinite(tod)):
            return np.zeros(tod.size)
        if tod.size < medfilt_stepsize:
            return np.zeros(tod.size) + np.nanmedian(tod)
        filter_tod = np.array(medfilt.medfilt(tod.astype(np.float64),np.int32(medfilt_stepsize)))
        
        return filter_tod[:tod.size]


    def frequency_spectra_per_band(self, tod):
        """
        Calculate the frequency spectra per band
        """
        n_bands, n_channels, n_tod = tod.shape
        tod_fft = np.zeros((n_bands, n_channels, n_tod),dtype=np.float64) + np.nan
        for iband in range(n_bands):
            good_channels = (tod[iband,:,0] != 0) & np.isfinite(tod[iband,:,0])
            if np.sum(good_channels) < 2:
                print("No good channels in band {}".format(iband)) 

                continue
            tod_fft[iband,good_channels,:] = np.abs(np.fft.fft(tod[iband,good_channels,...],axis=0))**2

        tod_fft = np.nanmean(tod_fft,axis=-1) 
        freqs = np.fft.fftfreq(n_channels, d=2e3/1024.)
        N = freqs.size//2

        tod_fft[~np.isfinite(tod_fft)] = 0
        power_spectra = [] 
        for iband in range(n_bands):
            power_spectra.append(FitPowerSpectrum()) 
            power_spectra[iband](freqs[10:N],tod_fft[iband,10:N],error_func=power_spectra[iband].log_error,
                                 model=power_spectra[iband].knee_frequency_model, min_freq=8e-3, max_freq=0.1,
                                 P0=[np.nanmedian(tod_fft[iband,10:N])**0.5, 1e-3, -1]) 
        return power_spectra

    def remove_atmosphere(self, airmass, tod, atmosphere_fit_values, source_name = ''):
        """
        Remove the atmosphere from the TOD
        """

        if source_name in Coordinates.CalibratorList: 
            return tod[...] - np.nanmedian(tod[...],axis=-1)[...,None]
        
        n_bands, n_channels, n_tod = tod.shape
        tod_clean = np.zeros((tod.shape[0],tod.shape[1],n_tod))
        for iband in range(n_bands):
            tod_clean[iband,:] = AtmosphereRemoval.subtract_fitted_atmosphere(airmass,
                                                                              tod[iband,...],
                                                                              atmosphere_fit_values[iband])
        return tod_clean 

    def fill_bad_data(self, tod):
        """Fill nan values with median of the tod"""
        n_bands, n_channels, n_tod = tod.shape
        data_reshape = tod[...].reshape((n_bands*n_channels, n_tod))
        nan_tod = np.isnan(data_reshape) 
        ones = np.ones(data_reshape.shape)*np.nanmedian(data_reshape,axis=1)[:,None] 
        data_reshape[nan_tod] = ones[nan_tod]
        return  data_reshape.reshape((n_bands, n_channels, n_tod))

    def normalise_data(self, tod, system_temperature, gains):
        """
        Normalise the data by the system temperature and gains
        """
        dv = 2e9/1024. 
        tau = 1./50.  
        N4 = tod.shape[-1]//4 * 4 
        index_1 = np.arange(0,N4,4)
        index_2 = np.arange(2,N4,4)
        diff = tod[...,index_1] - tod[...,index_2] 
        rms = np.nanstd(diff, axis=-1)/np.sqrt(2) * np.sqrt(dv * tau) 

        return tod/rms[...,np.newaxis], rms[...,np.newaxis] 

    def median_filter(self, tod, medfilt_stepsize : int):
        """
        Calculate this AFTER removing the atmosphere.
        """
        n_bands, n_channels, n_tod = tod.shape
        filtered_tod = np.zeros((n_bands, n_channels, n_tod))
        for iband in tqdm(range(n_bands)):
            index = np.arange(1024,dtype=int)
            index = index[10:-10]
            index = index[(index < 512-5) | (index > 512+5)]
            masked_tod = tod[iband,index,:]
            mean_tod = np.nanmean(masked_tod,axis=0)
            if np.nansum(np.isfinite(mean_tod)) < medfilt_stepsize*2:
                continue 

            pad_tod = np.zeros(n_tod*3) 
            pad_tod[:n_tod] = mean_tod[::-1]
            pad_tod[n_tod:2*n_tod] = mean_tod
            pad_tod[2*n_tod:] = mean_tod[::-1] 
            median_filter_tod = medfilt.medfilt(pad_tod.astype(np.float64), medfilt_stepsize)[n_tod:2*n_tod]
            A = np.ones((n_tod,2))
            A[:,1] = median_filter_tod 
            x = np.linalg.solve(np.dot(A.T,A), np.dot(A.T,masked_tod.T)) 

            filtered_tod[iband,index] = masked_tod - (A.dot(x)).T

        
        return filtered_tod

    def gain_subtraction(self,data, tod, system_temperature):
        """Wrapper for the gain subtraction fit
        
        We calculate the power spectrum of the data first to use a prior on the gain fluctuations
        (Currently not in use)

        """
        ps_fits = self.fit_power_spectrum(data, tod[0]) 

        if not data.source_name in Coordinates.CalibratorList: 
            gain_function = getattr(GainSubtraction, self.gain_subtraction_name)
            dG = gain_function(tod, system_temperature, ps_fits) 
        else:
            logging.debug(f'{self.name} Calibrator observation ({data.source_name}). Not applying gain correction.')
            dG = None
        return dG 

    def plot_gain_examples(self, data, tod, avg_tod, dG, level2_data, iscan, ifeed,tsys, figure_path='figures/'):
        """ """
        figure_directory = f'{figure_path}/{data.obsid}/'
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        freq_min = 26
        freq_max = 28
        time_min = 0
        time_max = dG.size / 50. 

        # First plot the tod as an image
        frequencies = data.frequency[0,:]
        rms=  np.nanstd(tod,axis=1) 
        tsys_norm = tsys/np.nanmedian(tsys)
        rms_norm = rms/np.nanmedian(rms)
        fig, ax = pyplot.subplots(1,1,figsize=(10,10))
        pyplot.plot(frequencies,tsys_norm, label='Normalised Tsys')
        pyplot.plot(frequencies,rms_norm, label='Normalised RMS')
        pyplot.xlabel("Frequency [GHz]")
        pyplot.ylabel("Normalised Units")
        pyplot.title("Tsys and Tsys RMS for Scan {:02d} Feed {:02d}".format(iscan, ifeed))
        pyplot.legend()
        pyplot.ylim(0.5,2)
        fig.savefig(f'{figure_directory}/tsys_rms_scan{iscan:02d}_feed{ifeed:02d}.png')
        pyplot.close(fig)

        fig, ax = pyplot.subplots(1,1,figsize=(10,10))
        mad = np.nanmedian(np.abs(tod[...]-np.nanmedian(tod[...])))
        vmin, vmax = -2*mad, 2*mad 

        tsys = np.zeros_like(tod)
        tsys += self.level2.system_temperature[0,0,0,:,None]
        ax.imshow(tod[...], aspect='auto', origin='lower', vmin=vmin, vmax=vmax, extent=[time_min, time_max, freq_max, freq_min])
        pyplot.axhline(26.9, color='r', linestyle='--')
        pyplot.axhline(26.98, color='r', linestyle='--')
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title("Time ordered data Scan {} Feed {}".format(iscan, ifeed))
        fig.savefig(f'{figure_directory}/tod_scan_{iscan:02d}_feed_{ifeed:02d}.png')
        pyplot.close(fig)

        # Now plot weighted average of the tod with gain overplotted
        fig, ax = pyplot.subplots(2,1,figsize=(10,10))
        ax[0].plot(avg_tod[...], label="Weighted average")
        ax[0].plot(gaussian_filter1d(dG[...],5), label="Gain")
        ax[0].set_xlabel("Sample")
        ax[0].set_ylabel("Normalised Units")
        ax[0].set_title("Time ordered data")
        ax[0].legend()

        tsys_spike = np.where((data.frequency[0] > 26.9) & (data.frequency[0] < 26.98))[0]
        tod_slice = np.nanmean(tod[:,tsys_spike],axis=1)
        ax[1].plot(tod_slice, label="Tsys Spike TOD")
        ax[1].set_xlabel('Sample')
        ax[1].set_ylabel('Normalised Units')
        ax[1].set_title("Time ordered data of Tsys Spike")
        
        pyplot.suptitle("Scan {} Feed {}".format(iscan, ifeed))

        pyplot.tight_layout()
        fig.savefig(f'{figure_directory}/tod_scan_{iscan:02d}_feed_{ifeed:02d}_avg.png')
        pyplot.close(fig)
            

    def average_tod(self, data : HDF5Data, level2_data : COMAPLevel2) -> HDF5Data:
        """Average the time ordered data in frequency
        
        Overview:
        ---------
        Outer loop is over feeds and inner loop is over "scans" (a scan is defined as a continuous period of time where the telescope is pointing at the same az/el)

        
        """
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        n_channels_low = 4 # n_channels//self.frequency_bin_size 

        bandwidth = (2e9/n_channels) 
        sample_rate = 50. 
        self.tod_cleaned  = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_original = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_weights  = np.zeros((n_feeds, n_channels_low, n_tod))
        self.scan_edges   = data.scan_edges
        n_scans = len(self.scan_edges) 
        self.freq_power_spectra = np.zeros((n_scans, n_feeds, n_bands, 15,2))
        self.freq_power_spectra_fits = np.zeros((n_scans, n_feeds, n_bands, 3))
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='TOD Averaging Loop'):
            if (feed > 19):
                continue
            tod = data['spectrometer/tod'][ifeed, ...]

            for iscan, (start, end) in enumerate(self.scan_edges): 
                logging.debug(f'{self.name}: Averaging level 1 {iscan:02d} in Feed {feed:02d}')

                tod[...,start:end] = self.fill_bad_data(tod[...,start:end])

                # First we remove the atmospheric fluctuations using the sky dip measurements
                clean_tod = self.remove_atmosphere(data.airmass[ifeed,start:end], tod[...,start:end], level2_data['atmosphere/fit_values'][iscan,ifeed,:], source_name=data.source_name)

                # Second we normalise the data by the system temperature and gains
                clean_tod, normalisation_factor = self.normalise_data(clean_tod, level2_data['vane/system_temperature'][0,ifeed,:,:], level2_data['vane/system_gain'][0,ifeed,:,:]) 

                # Third we apply a high-pass median filter to remove the worst large-scale noise 
                clean_tod = self.median_filter(clean_tod, int(50*120))
                try:
                    # Here we calculate the relative gain fluctuations
                    dG = self.gain_subtraction(data, clean_tod, level2_data['vane/system_temperature'][0,ifeed,:,:]) 
                except (ValueError, IndexError): 
                    dG = None

                # Build weights, mask out edge channels and centre channels
                weights = 1./level2_data.system_temperature[0,ifeed]**2 
                weights[level2_data.system_temperature[0,ifeed] == 0] = 0
                weights[:,:10] = 0
                weights[:,-10:] = 0
                weights[:,510:515] = 0
                if not isinstance(dG, type(None)):
                    print('AFTER GAIN', np.nanstd(dG))

                if not isinstance(dG, type(None)):
                    residual = (clean_tod - dG[None,None,:])*normalisation_factor/level2_data['vane/system_gain'][0,ifeed,:,:,None]
                else:
                    residual = clean_tod*normalisation_factor/level2_data['vane/system_gain'][0,ifeed,:,:,None]

                residual = self.weighted_average_over_band(residual, weights) 

                if (iscan == 0) & (not isinstance(dG, type(None))):
                    avg_tod = self.weighted_average_over_band(clean_tod, weights) 
                    ### Plot some diagnostics
                    self.plot_gain_examples(data, clean_tod[0], avg_tod[0], dG, level2_data, iscan, feed,
                            level2_data.system_temperature[0,ifeed,0,:,None],figure_path=self.figure_directory)
                    ### 
                clean_tod = clean_tod*level2_data.system_temperature[0,ifeed,:,:,None]
                avg_tod = self.weighted_average_over_band(clean_tod, weights) 

                self.tod_cleaned[ifeed,:,start:end]  = residual
                self.tod_original[ifeed,:,start:end] = avg_tod
                self.tod_weights[ifeed,:,start:end]  = 1./self.auto_rms(residual)[:,None]**2

                #for iband in range(n_bands):
                #    self.freq_power_spectra[iscan,ifeed, iband, :,1] = frequency_power_spectra[iband].P_bin 
                #    self.freq_power_spectra[iscan,ifeed, iband, :,0] = frequency_power_spectra[iband].nu_bin 
                #    self.freq_power_spectra_fits[iscan,ifeed, iband, :] = frequency_power_spectra[iband].result.x

@dataclass 
class Level1Plotting(PipelineFunction):    
    name : str = 'Level1Plotting'

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    figure_directory : str = 'figures'     
    _full_figure_directory : str = 'figures' # Appends observation id in __call__ function

    SAMPLE_RATE : float = 50. # Hz 
    N_CHANNELS : int = 1024
    STATE : bool = True 

    def __call__(self, data : HDF5Data) -> HDF5Data:
                
        self._full_figure_directory = f'{self.figure_directory}/{data.obsid}'
        if not os.path.exists(self._full_figure_directory):
            os.makedirs(self._full_figure_directory)

        self.plot_vanes(data)
        self.plot_frequency_spectra(data)

        return self.STATE
    
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {}
        attrs = {}

        return data, attrs

    def plot_vanes(self, data : HDF5Data): 
        """Plot the system temperature and gain"""

        n_feeds, n_bands, n_channels, n_tod = data.tod_shape
        n_scans = len(data.scan_edges) 

        fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)
        ax[0].set_title(f'Vane Obs: {data.observation_id} Source: {data.source_name}')
        ax[0].set_ylabel('System Temperature [K]')
        ax[1].set_ylabel('System Gain [V/K]')
        ax[1].set_xlabel('Time [s]')
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='Vane Plotting Loop'):
            ax[0].plot(data['spectrometer/frequency'].flatten(), data['vane/system_temperature'][0,ifeed,:,:].flatten(), label=f'Feed {ifeed}')
            ax[1].plot(data['spectrometer/frequency'].flatten(), data['vane/system_gain'][0,ifeed,:,:].flatten(), label=f'Feed {ifeed}')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f'{self._full_figure_directory}/vane_feed_{feed:02d}.png')
        plt.close(fig)

    def plot_frequency_spectra(self, data : HDF5Data): 
        """Plot frequency spectra generated in Level1AveragingGainCorrection"""

        n_feeds, n_bands, n_channels, n_tod = data.tod_shape 
        n_scans = len(data.scan_edges) 

        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='Frequency Spectra Plotting Loop'):
            fig, ax = plt.subplots(1,1, figsize=(12,8))
            ax[0].set_title(f'Frequency Spectra Feed: {feed:02d} Obs: {data.observation_id} Source: {data.source_name}')
            ax[0].set_ylabel('Power [K$^2$]')
            for iscan, (start, end) in enumerate(data.scan_edges): 
                for iband in range(n_bands):
                    ax[0].plot(self.level2.freq_power_spectra[iscan,ifeed, iband, :,0], self.level2.freq_power_spectra[iscan,ifeed, iband, :,1], label=f'Scan {iscan} Band {iband}')
            ax[0].legend()
            fig.tight_layout()
            fig.savefig(f'{self._full_figure_directory}/frequency_spectra_feed_{feed:02d}.png')
            plt.close(fig)


