#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:17:03 2023

AtmosphereRemoval - 
Level1Averaging -
Level1AveragingGainCorrection - 

@author: sharper
"""

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 
from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2

@dataclass 
class AtmosphereRemoval(PipelineFunction):    
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data  = {}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data) -> HDF5Data:
                
        self.filter_atmosphere(data)
        
        return data

    def filter_atmosphere(self, data : HDF5Data):
        
        A = 1./np.sin(data.el*np.pi/180.)
        n_feeds, n_bands, n_channels, n_tod = data.shape 

        for (ifeed, feed) in tqdm(data.tod_loop(bands=False, channels=False), desc='Atmosphere Filter Feed Loop'):
            tod = data['spectrometer/tod'][ifeed,...]
            tod_mean = np.nanmean(tod,axis=-1)
            tod = tod/tod_mean[...,None] - 1

        Z = np.ones((A.size, 2))
        Z[:,1] = A
        Z = block_diag([Z]*Nfreq)
        Zmasked = np.ones((Amasked.size, 2))
        Zmasked[:,1] = Amasked
        Zmasked = block_diag([Zmasked]*Nfreq)
        
        
        # Remove the atmosphere fluctuations
        if not any([feature==5.0 for feature in features]):
            tod_ps = np.zeros(tod.shape)
            for i in range(tod.shape[0]):
                d = tod[i,lowf:highf,~src_mask].flatten()[:,None]
                b = Zmasked.T.dot(d)
                M = (Zmasked.T.dot(Zmasked))
                a = linalg.spsolve(M,b)
                tod_ps[i] = tod[i,lowf:highf,:] - np.reshape(Z.dot(a[:,None]),(Nfreq,tod.shape[2]))
        else:
            tod_ps = tod[:,lowf:highf,:]

@dataclass 
class Level1Averaging(PipelineFunction):    
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 
    
    tod : np.ndarray = field(default_factory=lambda : np.zeros(1))
    tod_stddev : np.ndarray = field(default_factory=lambda : np.zeros(1))

    frequency_mask : np.ndarray = field(default_factory=lambda : np.zeros(1,dtype=bool))

    frequency_bin_size : int = 512 
    
    N_CHANNELS : int = 1024

    def __post_init__(self):
        
        self.frequency_mask = np.zeros(self.N_CHANNELS,dtype=bool)
        self.frequency_mask[:10] = True
        self.frequency_mask[-10:]= True 
        self.frequency_mask[511:514] = True
        
        self.channel_edges = np.arange(self.N_CHANNELS//self.frequency_bin_size + 1)
        self.channel_idx = np.arange(self.N_CHANNELS)

    def __call__(self, data : HDF5Data) -> HDF5Data:
                
        self.average_tod(data)
        
        return data
    
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
class Level1AveragingGainCorrection(Level1Averaging):
    

    def average_tod(self, data : HDF5Data) -> HDF5Data:
        """Average the time ordered data in frequency"""
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        n_channels_low = n_channels//self.frequency_bin_size 

        self.tod = np.zeros((n_feeds, n_bands, n_channels_low, n_tod))
        self.tod_stddev = np.zeros((n_feeds, n_bands, n_channels_low, n_tod))

        for (ifeed, feed), iband in tqdm(data.tod_loop(channels=False), desc='TOD Averaging Loop'):
            tod = data['spectrometer/tod'][ifeed, iband, ...]
            
            for iscan, (start, end) in enumerate(scan_edges): 
                
                
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

