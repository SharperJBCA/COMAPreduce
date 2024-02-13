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
from matplotlib import pyplot 
import h5py

from dataclasses import dataclass, field 

from .Running import PipelineFunction
from .PowerSpectra import FitPowerSpectrum
from .DataHandling import HDF5Data , COMAPLevel2
from comancpipeline.Tools.stats import auto_rms
from comancpipeline.Tools import Coordinates
from scipy.signal import find_peaks, peak_widths

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
class UseLevel2Pointing(PipelineFunction):
    """Class was implemented to test the pointing updates """
    name : str = 'UseLevel2Pointing'
    
    overwrite : bool = False

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2) 

    def __post_init__(self):
        
        self.data = {}
        self.attrs = {} 
        self.groups = list(self.data.keys())
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
            
        attrs = {}

        return self.data, self.attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2):
        
        if not self.overwrite:
            return self.STATE
        if not os.path.exists(level2_data.filename):
            return self.STATE 
        h = h5py.File(level2_data.filename,'r')
        level2_data.ra = h['spectrometer/pixel_pointing/pixel_ra'][...]
        level2_data.dec = h['spectrometer/pixel_pointing/pixel_dec'][...]
        level2_data.az = h['spectrometer/pixel_pointing/pixel_az'][...]
        level2_data.el = h['spectrometer/pixel_pointing/pixel_el'][...]
        h.close()

        data.ra = level2_data.ra
        data.dec= level2_data.dec
        data.az = level2_data.az
        data.el = level2_data.el
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

    def __call__(self, filelist : list):
                
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

    def plot_mean_vanes(self, filelist : list): 
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
    groups : list = field(default_factory=lambda: ['fnoise_fits'])

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
        if data.source_name in Coordinates.CalibratorList: 
            return self.STATE
        
        self._full_figure_directory = f'{self.figure_directory}/{data.obsid}'
        if not os.path.exists(self._full_figure_directory):
            os.makedirs(self._full_figure_directory)

        self.run(data, level2_data)

        return self.STATE

    def run(self, data : HDF5Data, level2_data : COMAPLevel2):

        N_SCANS = len(data.scan_edges)
        self.data = {'fnoise_fits/fnoise_fit_parameters':np.zeros((self.N_FEEDS, self.N_BANDS, N_SCANS, 3)),
                     'fnoise_fits/auto_rms':np.zeros((self.N_FEEDS, self.N_BANDS, N_SCANS))}
        for ((ifeed, feed),iband) in tqdm(level2_data.tod_loop(bands=True)):
            if (feed > 19):
                continue
            _tod = level2_data['averaged_tod/tod'][ifeed,iband]

            for iscan, (start, end) in enumerate(data.scan_edges): 
                tod = _tod[start:end] 
                if np.nansum(tod) == 0:
                    continue
                power_spectrum = np.abs(np.fft.fft(tod))**2/tod.size 
                freqs = np.fft.fftfreq(len(tod), d=1./self.SAMPLE_RATE) 
                power_spectrum = power_spectrum[freqs > 0]
                freqs = freqs[freqs > 0]
                diff_tod = np.diff(tod)
                auto_rms = np.nanstd(diff_tod)/np.sqrt(2) 
    
                mask = np.ones(freqs.size, dtype=bool)
                mad = np.median(np.abs(power_spectrum[freqs > 1] - np.median(power_spectrum[freqs > 1])))*1.4826
                niter = 3
                indices = np.arange(freqs.size, dtype=int)
                for istep in range(niter):
                    select = mask & (freqs > 0.5)
                    peak_idx, properties = find_peaks(power_spectrum[select], height=auto_rms**2*100, distance=100)
                    peak_idx = (indices[select])[peak_idx]
                    widths, width_heights, left_ips, right_ips = peak_widths(power_spectrum, peak_idx, rel_height=0.85)
                    for i in range(len(peak_idx)):
                        mask[int(left_ips[i]):int(right_ips[i])] = False

                # plot the power spectrum
                try:
                    fig, ax = pyplot.subplots(1,1, figsize=(12,8))
                    ax.plot(freqs, power_spectrum)
                    ax.plot(freqs[~mask], power_spectrum[~mask], '.',color='red')
                    ax.axhline(auto_rms**2, color='k', linestyle='--')
                    ax.axhline(100*auto_rms**2, color='k', linestyle=':')
                    ax.set_title(f'Feed: {ifeed:02d} Band: {iband:02d}')
                    ax.set_xlabel('Frequency [Hz]')
                    ax.set_ylabel('Power Spectrum')
                    pyplot.yscale('log')
                    pyplot.xscale('log')
                    fig.tight_layout()
                    fig.savefig(f'{self._full_figure_directory}/feed_{feed:02d}_band_{iband:02d}_scan_{iscan:02d}_power_spectrum_full.png')
                    pyplot.close(fig)
                except ValueError:
                    print('\n#####',data.obsid,tod.shape,np.nansum(tod),np.nansum(power_spectrum), ifeed, iband,'######\n')
                    raise(ValueError)

                ps = FitPowerSpectrum(nbins=30) 
                ps(freqs[mask], power_spectrum[mask], 
                    errors=None, 
                    model=ps.red_noise_model, 
                    error_func=ps.log_error, P0=None,
                    min_freq=0.05)

                if not isinstance(ps.result, type(None)):
                    ps.plot_fit(fig_dir=self._full_figure_directory, prefix = f'feed_{feed:02d}_band_{iband:02d}_scan_{iscan:02d}')
                    self.data['fnoise_fits/fnoise_fit_parameters'][ifeed,iband,iscan] = ps.result.x
                    self.data['fnoise_fits/auto_rms'][ifeed,iband,iscan] = auto_rms