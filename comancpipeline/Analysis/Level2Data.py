#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:47:59 2023

@author: sharper
"""
from os import path, makedirs
import numpy as np
from tqdm import tqdm
from astropy.time import Time 
import os

from dataclasses import dataclass, field 

from .Running import PipelineFunction
from .PowerSpectra import FitPowerSpectrum
from .DataHandling import HDF5Data , COMAPLevel2
from comancpipeline.Tools.stats import auto_rms

@dataclass 
class AssignLevel1Data(PipelineFunction):
    name : str = 'AssignLevel1Data'
    
    overwrite : bool = False

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2) 

    def __post_init__(self):
        
        self.data = {'spectrometer/MJD':np.empty(1),
                'spectrometer/feeds':np.empty(1),
                'spectrometer/bands':np.empty(1),
                'spectrometer/features':np.empty(1),
                'spectrometer/frequency':np.empty(1),
                'spectrometer/pixel_pointing/pixel_ra':np.empty(1),
                'spectrometer/pixel_pointing/pixel_dec':np.empty(1),
                'spectrometer/pixel_pointing/pixel_az':np.empty(1),
                'spectrometer/pixel_pointing/pixel_el':np.empty(1)}
        self.attrs = {} 
        self.groups = list(self.data.keys())
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
            
        attrs = {}

        return self.data, self.attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2):
        
        self.data['spectrometer/MJD']   = data['spectrometer/MJD']
        self.data['spectrometer/feeds'] = data['spectrometer/feeds']
        self.data['spectrometer/bands'] = data['spectrometer/bands']
        self.data['spectrometer/features']=data.features 
        self.data['spectrometer/frequency']=data['spectrometer/bands']
        self.data['spectrometer/pixel_pointing/pixel_ra'] = data['spectrometer/pixel_pointing/pixel_ra']
        self.data['spectrometer/pixel_pointing/pixel_dec']= data['spectrometer/pixel_pointing/pixel_dec']
        self.data['spectrometer/pixel_pointing/pixel_az'] = data['spectrometer/pixel_pointing/pixel_az']
        self.data['spectrometer/pixel_pointing/pixel_el'] = data['spectrometer/pixel_pointing/pixel_el']
        
        self.attrs['comap'] = {k:v for k,v in data.attrs('comap').items()}
        
        return self.STATE

@dataclass 
class WriteLevel2Data(PipelineFunction):
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2) 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data) -> HDF5Data:

        del data 
        self.write_level2_file()
        
        return self.level2 
    
    def write_level2_file(self):
        """ """ 
        
        data_dir = path.dirname(self.level2.filename)
        if not path.exists(data_dir):
            makedirs(data_dir)
            
        self.level2.write_data_file(self.level2.filename) 

@dataclass 
class Level2Timelines(PipelineFunction):    
    # A class to plot stats over time for a set of observations 

    name : str = 'Level2Timelines'

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    source = 'none' 
    figure_directory : str = 'figures'     
    _full_figure_directory : str = 'figures' # Appends observation id in __call__ function

    N_FEEDS : int = 20 
    N_BANDS : int = 4
    SAMPLE_RATE : float = 50. # Hz 
    N_CHANNELS : int = 1024
    STATE : bool = True 

    def __call__(self, filelist : list[str]):
                
        self._full_figure_directory = f'{self.figure_directory}/{data.obsid}'
        if not os.path.exists(self._full_figure_directory):
            os.makedirs(self._full_figure_directory)

        self.plot_mean_vanes(filelist)

        return self.STATE
    
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {}
        attrs = {}

        return data, attrs

    def plot_mean_vanes(self, filelist : list[str]): 
        """Plot median system temperature and gain over time"""

        # First gather data for plotting from all of the files
        n_obs = len(filelist)
        tsys = np.zeros((n_obs, self.N_FEEDS, self.N_BANDS))
        gain = np.zeros((n_obs, self.N_FEEDS, self.N_BANDS))
        mjd  = np.zeros((n_obs))
        for ifile,filename in enumerate(tqdm(filelist)):
            data = COMAPLevel2(large_datasets=['averaged_tod/tod', 'averaged_tod/weights', 'averaged_tod/tod_original'])
            data.read_data_file(filename) 
            source_name = data.source_name
            if not source_name == self.source:
                del data 
                continue
            mjd[ifile]  = data['spectrometer/MJD'][0]
            tsys[ifile] = system_temperature.median(axis=1)
            gain[ifile] = system_gain.median(axis=1)
            del data 

        # Remove any zeros from the data
        tsys = tsys[mjd != 0]
        gain = gain[mjd != 0]
        mjd = mjd[mjd != 0]

        # Now plot the data
        # convert MJD to datetime, format datetime as YY-MM-DD 
        mjd = Time(mjd, format='mjd')
        mjd.format = 'datetime'
        mjd = mjd.value
        mjd = [mjd[i].strftime('%Y-%m-%d') for i in range(len(mjd))]
        fig, ax = plt.subplots(2,1, figsize=(12,8))
        ax[0].set_title(f'Median System Source: {source_name}')
        ax[0].set_ylabel('System Temperature [K]')
        ax[0].set_xlabel('Time [YY-MM-DD]')
        ax[1].set_ylabel('System Gain')
        ax[1].set_xlabel('Time [YY-MM-DD]')

        for ifeed in range(self.N_FEEDS):
            for iband in range(self.N_BANDS):
                ax[0].plot(mjd, tsys[:,ifeed,iband], label=f'Feed: {ifeed:02d} Band: {iband}')
                ax[1].plot(mjd, gain[:,ifeed,iband], label=f'Feed: {ifeed:02d} Band: {iband}')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(f'{self._full_figure_directory}/median_system_temperature_gain.png')

@dataclass 
class Level2FitPowerSpectrum(PipelineFunction):

    # A class to plot stats over time for a set of observations 

    name : str = 'Level2FitPowerSpectrum'

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 
    groups : list[str] = field(default_factory=lambda: ['fnoise_fits'])

    source = 'none' 
    figure_directory : str = 'figures'     
    _full_figure_directory : str = 'figures' # Appends observation id in __call__ function

    N_FEEDS : int = 20 
    N_BANDS : int = 4
    SAMPLE_RATE : float = 50. # Hz 
    N_CHANNELS : int = 1024
    STATE : bool = True 

    overwrite : bool = False
    data : dict = field(default_factory=dict)
    
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        attrs = {}

        return self.data, attrs


    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
        self._full_figure_directory = f'{self.figure_directory}/{data.obsid}'
        if not os.path.exists(self._full_figure_directory):
            os.makedirs(self._full_figure_directory)
        self.run(data, level2_data)

        return self.STATE

    def run(self, data : HDF5Data, level2_data : COMAPLevel2):

        self.data = {'fnoise_fits/fnoise_fit_parameters':np.zeros((self.N_FEEDS, self.N_BANDS, 3))}
        for ((ifeed, feed),iband) in tqdm(level2_data.tod_loop(bands=True)):
            if (feed > 19):
                continue
            tod = level2_data['averaged_tod/tod'][ifeed,iband]
            power_spectrum = np.abs(np.fft.fft(tod))**2
            freqs = np.fft.fftfreq(len(tod), d=1./self.SAMPLE_RATE) 
            power_specturm = power_spectrum[freqs > 0]
            freqs = freqs[freqs > 0]

            ps = FitPowerSpectrum(nbins=30) 
            ps(freqs, power_specturm, 
                errors=None, 
                model=ps.red_noise_model, 
                error_func=ps.log_error, P0=None,
                min_freq=0.5)

            if not isinstance(ps.result, type(None)):
                ps.plot_fit(fig_dir=self._full_figure_directory, prefix = f'feed_{ifeed:02d}_band_{iband:02d}_')
                self.data['fnoise_fits/fnoise_fit_parameters'][ifeed,iband] = ps.result.x