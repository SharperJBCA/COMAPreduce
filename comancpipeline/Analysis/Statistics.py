#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:18:15 2023

Classes for running statistics on level 2 files.

@author: sharper
"""

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 
from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2
from scipy.sparse import linalg, block_diag
from comancpipeline.Tools.median_filter import medfilt

from scipy.optimize import minimize
import logging 
import warnings
from comancpipeline.Tools import Coordinates 
from matplotlib import pyplot
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass 
class Spikes(PipelineFunction):    
    
    name : str = 'Spikes'
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    overwrite : bool = False 
    STATE : bool = True 
    
    MEDIAN_FILTER_STEP : int = 100
    SPIKE_THRESHOLD : float = 10
    def __post_init__(self):
        """Create the save data structure"""
        
        self.data = {'spikes/spike_mask':np.empty(1)}

        self.groups = np.unique([s.split('/')[0] for s in self.data.keys()])
        
    def parameter_template(self, k):
        return self.data[k] 
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        attrs = {}
        return self.data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
        if not data.source_name in Coordinates.CalibratorList:
            self.run_fit_spikes(data, level2_data)
        
        return self.STATE
    
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

    def fit_spikes(self, tod : np.ndarray[float], rms : float, step : int = 100):
        
        tod_clean = tod - self.median_filter(tod, self.MEDIAN_FILTER_STEP)
        mask = (np.abs(tod_clean) > rms*self.SPIKE_THRESHOLD) 
        diff = np.diff(mask.astype(float))
        starts = np.where((diff > 0))[0]
        ends = np.where((diff < 0))[0]
        
        if mask[0]: 
            starts = np.insert(starts,0,0) 
        if mask[-1]: 
            ends = np.append(ends,tod_clean.size) 

        for (start,end) in zip(starts, ends):
            s = int(max([0,start-step]))
            e = int(min([tod.size,end+step]))
            mask[s:e] = True

        return mask
    
    def run_fit_spikes(self, data : HDF5Data, level2_data : COMAPLevel2): 
        
        n_feeds, n_bands, n_tod = level2_data.tod_shape 
        self.data['spikes/spike_mask'] = np.zeros((n_feeds, n_bands, n_tod), dtype=bool)
        for ((ifeed, feed),iband) in tqdm(level2_data.tod_loop(bands=True), desc='Fitting Spikes'):
            rms = level2_data.tod_auto_rms(ifeed, iband) 
            for iscan, (start,end) in enumerate(level2_data.scan_edges):
                tod = level2_data.tod[ifeed,iband,start:end] 
                self.data['spikes/spike_mask'][ifeed,iband, start:end] =  self.fit_spikes(tod, rms) 
  
@dataclass 
class NoiseStatistics(PipelineFunction):    
    
    name : str = 'NoiseStatistics'
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    overwrite : bool = False 
    STATE : bool = True 
    N_FN_PARAMETERS : int = 3
    
    def __post_init__(self):
        """Create the save data structure"""
        
        self.data = {'noise_statistics/fnoise':np.empty(1),
                     'noise_statistics/auto_rms':np.empty(1)}
        self.groups = np.unique([s.split('/')[0] for s in self.data.keys()])
        

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        attrs = {}

        return self.data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
        if not data.source_name in Coordinates.CalibratorList:
            self.run_fit_noise(data, level2_data)
        
        return self.STATE
    
    @staticmethod
    def model(P,x):
        """
        Assuming model \sigma_w^2 + \sigma_r^2 (frequency/frequency_r)^\alpha
        Parameters
        ----------
        Returns
        -------
        
        """
        return P[0] + P[1]*np.abs(x/0.1)**P[2]
    
    @staticmethod
    def power_spectrum(tod : np.ndarray[float], sample_rate : float=1./50., nbins : int =15):
        """Creates binned power spectrum"""
        
        ps = np.abs(np.fft.fft(tod)**2)
        ps_nu = np.fft.fftfreq(ps.size,d=sample_rate)

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

        return nu_bin, P_bin

    def fit_power_spectrum(self,tod : np.ndarray[float], lowf=10, highf=-10):
        """ Fits the powerspectrum """

        def error(P,x,y,sig2, model):
        
            chi2 = np.sum((np.log(y)-np.log(model([sig2,P[0],P[1]],x)))**2)
            if not np.isfinite(chi2):
                return np.inf
            
            return chi2

        nu_bin, P_bin = self.power_spectrum(tod)        
        
        if len(nu_bin) == 0:
            return [np.nan, np.nan, np.nan]
        
        P0 = [P_bin[np.argmin((nu_bin-1)**2)], -1]
        gd = (nu_bin > 0.1) # just the high frequencies
        result = minimize(error,P0,args=(nu_bin[gd],P_bin[gd],P_bin[-1],self.model),
                          bounds=([0,None],[None,0]))
        
        results = [P_bin[-1], result.x[0],result.x[1]]
        return results

    def plot_fnoise(self,tod : np.ndarray[float], parameters : np.ndarray[float]):
        
        nu_bin, P_bin = self.power_spectrum(tod)        
        model = self.model(parameters,nu_bin) 
        fig,ax = pyplot.subplots()
        pyplot.plot(nu_bin, P_bin,'k')
        pyplot.plot(nu_bin, model,'C3',ls='--')
        pyplot.grid()
        pyplot.text(0.05,0.95,'\n'.join([f'{p:.2f}' for p in parameters]),va='top',ha='left',
                    transform=ax.transAxes)
        pyplot.xscale('log')
        pyplot.yscale('log')
        pyplot.show()
        
    def run_fit_noise(self,data : HDF5Data, level2_data : COMAPLevel2): 
        
        n_scans = len(level2_data.scan_edges)
        n_feeds, n_bands, n_tod = level2_data.tod_shape 
        self.data['noise_statistics/fnoise'] = np.zeros((n_feeds, n_bands, n_scans, self.N_FN_PARAMETERS))
        for ((ifeed, feed),iband) in tqdm(level2_data.tod_loop(bands=True), desc='Fitting 1/f'):
            for iscan, (start,end) in enumerate(level2_data.scan_edges):
                tod = level2_data.tod[ifeed,iband,start:end]*1.
                if 'spikes/spike_mask' in level2_data.keys(): 
                    spike_mask = level2_data['spikes/spike_mask'][ifeed,iband,start:end].astype(bool)
                    good = np.where(~spike_mask)[0] 
                    bad = np.where(spike_mask)[0] 
                    tod[spike_mask] = np.interp(bad,good,tod[~spike_mask])
                    
                self.data['noise_statistics/fnoise'][ifeed,iband,iscan] =  self.fit_power_spectrum(tod) 
