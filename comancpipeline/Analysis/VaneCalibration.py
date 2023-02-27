#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:22:15 2023

@author: sharper
"""

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 

from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2
from comancpipeline.Tools.stats import auto_rms

@dataclass 
class MeasureSystemTemperature(PipelineFunction):
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    system_temperature : np.ndarray = field(default_factory=lambda : np.zeros(1))
    system_gain : np.ndarray = field(default_factory=lambda : np.zeros(1))
        
    OBSID_MINIMUM : int = 7_000
    OBSID_MAXIMUM : int = 1_000_000 
    
    VANE_COLD_TEMP : float = 2.73 # K
    
    
    def __call__(self, data : HDF5Data) -> HDF5Data:
                
        self.measure_system_temperature(data)
        
        return data
    
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {'vane/system_temperature': self.system_temperature,
                'vane/system_gain': self.system_gain}
        attrs = {}

        return data, attrs
        
    def find_vane_samples(self, data : HDF5Data) -> (np.ndarray, int):
        """Find the vane calibration start/end samples"""
        
        vane_flag = data.vane_flag
        
        vane_indices = np.nonzero(np.diff(vane_flag))[0] + 1
        vane_indices = vane_indices.reshape((vane_indices.size//2,2))
        n_vanes = vane_indices.shape[0]
        
        return vane_indices, n_vanes
    
    def system_temperature_from_tod(self,
                                    vane_hot_temp : np.ndarray[float],
                                    tod : np.ndarray[float],  
                                    hot_samples : np.ndarray[int],  
                                    cold_samples : np.ndarray[int]) -> (np.ndarray, np.ndarray):
        """Measure the system temperature for a vane event"""
                
        temp_hot = np.nanmean(tod[...,hot_samples],axis=-1)
        temp_cold = np.nanmean(tod[...,cold_samples],axis=-1) 
        temp_diff = temp_hot - temp_cold
            
            
        gain = temp_diff/(vane_hot_temp - self.VANE_COLD_TEMP)
        tsys = temp_cold/gain
        
        return tsys, gain 
        
        
    def find_hot_cold_from_tod(self, tod : np.ndarray) -> (np.ndarray, np.ndarray):
        """Find the hot and cold samples of the vane event using the TOD"""
        
        def find_indices(vane_tod : np.ndarray, rms : float,
                         command: str='<', jump_size : int=50):
            """ """
            
            match command:
                case '>':
                    func = np.greater 
                case '<':
                    func = np.less
                
            mid_val = (np.nanmax(vane_tod)+np.nanmin(vane_tod))/2.
            tod_argsort = np.argsort(vane_tod)
            tod_sort    = np.sort(vane_tod)
            vane_group  = func(tod_sort-mid_val, rms)
            
            vane_tod = tod[(tod_argsort[vane_group])]
            X =  np.abs((vane_tod - np.median(vane_tod))/rms) < 1
            if np.sum(X) == 0:
                raise RuntimeError('No vane data found')

            id_vane = np.sort((tod_argsort[vane_group])[X])
            
            diff_cold = id_vane[1:] - id_vane[:-1]
            jumps = np.where((diff_cold > jump_size))[0]
            njumps = len(jumps)
            
            return id_vane, njumps
        
        rms = auto_rms(tod[:,None]).flatten()

        hot_id, hot_njumps  = find_indices(tod, rms, command='>')
        cold_id,cold_njumps = find_indices(tod, rms, command='<')

        hot_id  = np.sort(hot_id)
        cold_id = np.sort(cold_id)
        cold_id = cold_id[cold_id > min(hot_id)]
        
        return hot_id, cold_id

    def measure_system_temperature(self, data : HDF5Data):
        """Main loop over feeds/bands/channels"""
        
        vane_indices, n_vanes = self.find_vane_samples(data)
        
        # Setup the system temperature and gain data
        n_feeds, n_bands, n_channels, n_tod = data.tod_shape

        self.system_temperature = np.zeros((n_vanes, n_feeds, n_bands, n_channels))
        self.system_gain = np.zeros((n_vanes, n_feeds, n_bands, n_channels))

        for ivane, (start, end) in enumerate(vane_indices):
            for (ifeed, feed), iband in tqdm(data.tod_loop(channels=False), desc='System Temperature Loop'):
                tod = data['spectrometer/tod'][ifeed,iband,:,start:end]
                band_average = data['spectrometer/band_average'][ifeed,iband,start:end]

                hot_samples, cold_samples = self.find_hot_cold_from_tod(band_average)

                tsys, gain = self.system_temperature_from_tod(data.vane_temperature, tod, hot_samples, cold_samples) 

                self.system_temperature[ivane,ifeed,iband,:] = tsys
                self.system_gain[ivane,ifeed,iband,:] = gain
    