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

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 
from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2
from scipy.sparse import linalg, block_diag

from scipy.optimize import minimize
import logging 
import warnings
from comancpipeline.Tools import Coordinates 
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass 
class AtmosphereRemoval(PipelineFunction):    
    
    name : str = 'AtmosphereRemoval'
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    groups : list[str] = field(default_factory=lambda: ['atmosphere']) 

    overwrite : bool = False 
    STATE : bool = True 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {'atmosphere/fit_values':self.fit_values}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
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

    def fit_atmosphere(self, A, tod, lowf=10,highf=1024-10):
        """Fit for the atmosphere"""
        
        Nfreq = highf - lowf
        Zmasked = np.ones((A.size, 2))
        Zmasked[:,1] = A
        Zmasked = block_diag([Zmasked]*Nfreq)

        d = tod[lowf:highf,:].flatten()[:,None]
        
        b = Zmasked.T.dot(d)
        M = (Zmasked.T.dot(Zmasked))
        fit_values = linalg.spsolve(M,b)
        offset = np.zeros(1024) + np.nan
        atmos = np.zeros(1024) + np.nan
        offset[lowf:highf] = fit_values[::2] 
        atmos[lowf:highf] = fit_values[1::2]

        return offset, atmos
    
    def filter_atmosphere(self, data : HDF5Data):
        """ """
        A = data.airmass
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape 
        n_scans = len(data.scan_edges)
        logging.info(f'{self.name}: Total number of scans {n_scans:03d}')
        
        self.fit_values = np.zeros((n_scans, n_feeds, n_bands, 2, n_channels))
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='Atmosphere Filter Feed Loop'):
            _tod = data['spectrometer/tod'][ifeed,...]
            for iband in range(n_bands):
                tod = _tod[iband]
                for iscan, (start, end) in enumerate(data.scan_edges):
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
    groups : list[str] = field(default_factory=lambda: []) 
    
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
class Level1AveragingGainCorrection(Level1Averaging):
    name : str = 'Level1AveragingGainCorrection'
    groups : list[str] = field(default_factory=lambda: ['averaged_tod']) 
    
    overwrite : bool = False
    STATE : bool = True 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {'averaged_tod/tod':self.tod_cleaned,
                 'averaged_tod/tod_original':self.tod_original,
                 'averaged_tod/weights':self.tod_weights,
                 'averaged_tod/scan_edges':self.scan_edges}
                 
        attrs = {}

        return data, attrs
    
    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
        self.average_tod(data, level2_data)
        
        return self.STATE


    def auto_rms(self, tod : np.ndarray[float]):
        """ Calculate rms from differences of adjacent samples """ 
        
        N = tod.shape[-1]//2 * 2
        diff = tod[...,1:N:2] - tod[...,0:N:2]
        
        return np.nanstd(diff,axis=-1)/np.sqrt(2) 
        
    def normalised_data(self, data : HDF5Data, tod : np.ndarray[float]):
        """ """ 
        if (tod.shape[-1] == 0):
            return None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tod_mean = np.nanmean(tod, axis=2)
            tod_std  = self.auto_rms(tod) 
        
        tod_norm = (tod - tod_mean[...,None])/tod_std[...,None]
        
        return tod_norm
    
    def fit_power_spectrum(self, data : HDF5Data, tod : np.ndarray[float], lowf=10, highf=-10):
        """ Fits the averaged powerspectrum """
        def model(P,x):
            """
            Assuming model \sigma_w^2 + \sigma_r^2 (frequency/frequency_r)^\alpha
            Parameters
            ----------
            Returns
            -------
            
            """
            return P[0] + P[1]*np.abs(x/1.)**P[2]

        def error(P,x,y,sig2):
        
            chi2 = np.sum((np.log(y)-np.log(model([sig2,P[0],P[1]],x)))**2)
            if not np.isfinite(chi2):
                print(P) 
            
            return chi2
        resid_avg_orig = np.nanmean(tod[10:-10],axis=0)
        ps = np.abs(np.fft.fft(resid_avg_orig)**2)
        ps_nu = np.fft.fftfreq(ps.size,d=1./50.)

        nbins = 15
        nu_edges = np.logspace(np.log10(np.min(ps_nu[1:ps.size//2])),np.log10(np.max(ps_nu)),nbins+1)
        top = np.histogram(ps_nu,nu_edges,weights=ps)[0]
        bot = np.histogram(ps_nu,nu_edges)[0]
        gd = (bot != 0)
        P_bin = np.zeros(bot.size) + np.nan
        nu_bin = np.zeros(bot.size) + np.nan
        nu_bin[gd] = np.histogram(ps_nu,nu_edges,weights=ps_nu)[0][gd]/bot[gd]
        P_bin[gd] = top[gd]/bot[gd]
        
        gd = (bot != 0) & np.isfinite(P_bin) & (nu_bin != 0)
        nu_bin = nu_bin[gd]
        P_bin = P_bin[gd]
        
        
        if len(nu_bin) == 0:
            raise IndexError
        
        P0 = [P_bin[np.argmin((nu_bin-1)**2)], -1]
        gd = (nu_bin > 0.1) # just the high frequencies
        result = minimize(error,P0,args=(nu_bin[gd],P_bin[gd],P_bin[-1]),bounds=([0,None],[None,0]))
        
        results = [P_bin[-1], result.x[0],result.x[1]]
        return results
        
    def fit_gain_fluctuations(self, 
                              y_feed : np.ndarray[float,float],
                              tsys : np.ndarray[float], 
                              sigma0_prior : float, 
                              fknee_prior : float, 
                              alpha_prior : float):
        """
        Model: y(t, nu) = dg(t) + dT(t) / Tsys(nu) + alpha(t) / Tsys(nu) (nu - nu_0) / nu_0, nu_0 = 30 GHz
        """
        
        def model_prior(P,x):
            return P[1] * np.abs(x/1)**P[2]

        def gain_temp_sep(y, P, F, sigma0_g, fknee_g, alpha_g, samprate=50):
            
            freqs = np.fft.rfftfreq(len(y[0]), d=1.0/samprate)
            freqs[0] = freqs[1]
            n_freqs, n_tod = y.shape
            Cf = model_prior([sigma0_g**2, fknee_g, alpha_g], freqs)
            Cf[0] = 1
            
            N = y.shape[-1]//2 * 2
            sigma0_est = np.std(y[:,1:N:2] - y[:,0:N:2], axis=1)/np.sqrt(2)
            sigma0_est = np.mean(sigma0_est)
            Z = np.eye(n_freqs, n_freqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
            
            RHS = np.fft.rfft(F.T.dot(Z).dot(y))
        
            z = F.T.dot(Z).dot(F)
            a_bestfit_f = RHS/(z + 2*sigma0_est**2/Cf)
            a_bestfit = np.fft.irfft(a_bestfit_f, n=n_tod)
            from matplotlib import pyplot 
            print(a_bestfit)
            yz = y*1 
            yz[yz ==0] = np.nan 
            ym = np.nanmean(yz,axis=0)
            yz = np.nanmedian(yz,axis=0) 
            
            pmdl= np.poly1d(np.polyfit(np.nanmean(F*a_bestfit,axis=0).flatten(), ym, 1) )
            print(pmdl) 
            pyplot.plot(yz) 
            pyplot.plot(ym)
            pyplot.plot(np.nanmean(yz - F*a_bestfit,axis=0).flatten()*pmdl[1]) 
            pyplot.show()
            m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
            
            return a_bestfit, m_bestfit   
        
        nsb, Nfreqs, Ntod = y_feed.shape
        freqs = np.fft.rfftfreq(len(y_feed[0,30]), d=1.0/50.)
        RHS = np.abs(np.fft.rfft(y_feed[0,30]))**2 
        from matplotlib import pyplot
        pyplot.plot(freqs, RHS)
        pyplot.plot(freqs,model_prior([sigma0_prior**2, fknee_prior, alpha_prior], freqs))
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.show()


        scaled_freqs = np.linspace(-4.0 / 30, 4.0 / 30, 4 * 1024)  # (nu - nu_0) / nu_0
        scaled_freqs = scaled_freqs.reshape((4, 1024))
        scaled_freqs[(0, 2), :] = scaled_freqs[(0, 2), ::-1]  # take into account flipped sidebands
    
        P = np.zeros((4, Nfreqs, 2))
        F = np.zeros((4, Nfreqs, 1))
        P[:, :,0] = 1 / tsys
        P[:, :,1] = scaled_freqs/tsys
        F[:, :,0] = 1
    
        end_cut = 100
        # Remove edge frequencies and the bad middle frequency
        y_feed[:, :4] = 0
        y_feed[:, -end_cut:] = 0
        P[:, :4] = 0
        P[:, -end_cut:] = 0
        F[:, :4] = 0
        F[:, -end_cut:] = 0
        F[:, 512] = 0
        P[:, 512] = 0
        y_feed[:, 512] = 0
        y_feed[np.isnan(y_feed)] =0     
        
        # Reshape to flattened grid
        P = P.reshape((4 * Nfreqs, 2))
        F = F.reshape((4 * Nfreqs, 1))
        y_feed = y_feed.reshape((4 * Nfreqs, Ntod))
    
        # Fit dg, dT and alpha
        a_feed, m_feed = gain_temp_sep(y_feed, P, F, sigma0_prior, fknee_prior, alpha_prior)
        dg = a_feed[0]
        dT = m_feed[0]
        alpha = m_feed[1]
    
        return np.reshape(F*dg[None,:],(4,Nfreqs,dg.size)), dT, alpha
    
    def weighted_average_over_band(self, residual, weights):
        weights[:,:50] = 0 
        weights[:,-50:]= 0
        weights[:,512] = 0
        weights[np.isnan(residual[...,0])] =0 
        residual[np.isnan(residual)] =0 
        residual = np.sum(residual*weights[...,None],axis=1)/np.sum(weights[...,None],axis=1) 
        return residual
    
    def average_tod(self, data : HDF5Data, level2_data : COMAPLevel2) -> HDF5Data:
        """Average the time ordered data in frequency"""
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        n_channels_low = 4 # n_channels//self.frequency_bin_size 

        bandwidth = (2e9/n_channels) 
        sample_rate = 50. 
        self.tod_cleaned = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_original = np.zeros((n_feeds, n_channels_low, n_tod))
        self.tod_weights = np.zeros((n_feeds, n_channels_low, n_tod))
        self.scan_edges  = data.scan_edges
        for ((ifeed, feed),) in tqdm(data.tod_loop(bands=False, channels=False), desc='TOD Averaging Loop'):
            tod = data['spectrometer/tod'][ifeed, ...]
            
            for iscan, (start, end) in enumerate(self.scan_edges): 
                
                # Remove atmosphere 
                tod_clean = np.zeros((tod.shape[0],tod.shape[1],end-start))
                for iband in range(n_bands):
                    tod_clean[iband,:] = AtmosphereRemoval.subtract_fitted_atmosphere(data.airmass[ifeed,start:end],
                                                                              tod[iband,...,start:end],
                                                                              level2_data['atmosphere/fit_values'][iscan,ifeed,iband])
            
                # The normalise the data 
                tod_normed = self.normalised_data(data, tod_clean)/np.sqrt(bandwidth/sample_rate)
                
                tod_std  = self.auto_rms(tod_clean) 
                tsys = level2_data.system_temperature[0,ifeed,:,:].flatten()/np.sqrt(bandwidth/sample_rate)
                gain = level2_data.system_gain[0,ifeed,:,:].flatten()
                
                print(tod_normed.shape,np.sqrt(bandwidth/sample_rate))
                # There is a 3% difference between the Tsys and the auto_rms estimates
                correction_ratio = np.nanmedian((tsys*gain)/tod_std.flatten())

                if isinstance(tod_normed, type(None)):
                    logging.debug(f'{self.name}: {data.source_name} Feed {feed} no data in scan {iscan}') 
                    continue 
                
                logging.debug(f'{self.name}: {data.source_name} Feed {feed} shape of tod_normed {tod_normed.shape}; shape of tod {tod_clean.shape}')
                try:
                    ps_fits = self.fit_power_spectrum(data, tod_normed[0]) 
                except IndexError:
                    print(f'{rank} WARNING, FAILED PS_FITS:', tod_normed.shape, tod_clean.shape) 
                    continue
                # Finally, perform the gain fit
                
                if not data.source_name in Coordinates.CalibratorList: 
                    logging.debug(f'{self.name} Standard observation ({data.source_name}). Applying gain correction.')
                    dG, dT, alpha = self.fit_gain_fluctuations(tod_normed,
                                                               level2_data.system_temperature[0,ifeed], 
                                                               np.sqrt(ps_fits[0]*2), ps_fits[1], ps_fits[2])
                else:
                    logging.debug(f'{self.name} Calibrator observation ({data.source_name}). Not applying gain correction.')
                    dG = 0 
                      
                weights = 1./level2_data.system_temperature[0,ifeed]**2 
                weights[level2_data.system_temperature[0,ifeed] == 0] = 0 
                
                print(tod_normed.shape)
                from matplotlib import pyplot
                correction_ratio = 0.8
                pyplot.subplot(141)
                pyplot.imshow(tod_normed[0], aspect='auto',vmin=-0.7,vmax=0.7)
                pyplot.subplot(142)
                pyplot.imshow(dG[0]*correction_ratio, aspect='auto',vmin=-0.7,vmax=0.7)
                pyplot.subplot(143)
                pyplot.imshow(tod_normed[0] -  dG[0]*correction_ratio, aspect='auto',vmin=-0.7,vmax=0.7)
                pyplot.subplot(144)
                pyplot.plot(dT)

                pyplot.show()
                residual = (tod_normed - dG)*level2_data.system_temperature[0,ifeed,:,:,None]/np.sqrt(bandwidth/sample_rate)
                residual = self.weighted_average_over_band(residual, weights) 
                tod_normed = tod_normed*level2_data.system_temperature[0,ifeed,:,:,None]/np.sqrt(bandwidth/sample_rate)
                tod_normed = self.weighted_average_over_band(tod_normed, weights) 
                
                ps_clean = np.abs(np.fft.fft(residual[0]))**2
                ps_origi = np.abs(np.fft.fft(tod_normed[0]))**2
                nu = np.fft.fftfreq(tod_normed.shape[1], d=1./50.)
                from matplotlib import pyplot
                pyplot.plot(nu, ps_clean)
                pyplot.plot(nu, ps_origi)
                pyplot.xscale('log')
                pyplot.yscale('log')
                pyplot.show()

                self.tod_cleaned[ifeed,:,start:end]  = residual
                self.tod_original[ifeed,:,start:end] = tod_normed
                self.tod_weights[ifeed,:,start:end]  = 1./self.auto_rms(residual)[:,None]**2

        