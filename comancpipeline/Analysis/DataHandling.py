#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:05:24 2023

Carries all of the data between different parts of the pipeline

@author: sharper
"""
import os
import numpy as np
from h5py import File, Dataset
from dataclasses import dataclass, field
import itertools
from typing import Callable

    
@dataclass 
class HDF5Data:
    """General class for reading/writing in HDF5 files to memory using dictionaries"""
    
    large_datasets : list[str] = field(default_factory=list)
    hdf5_file : File = None
    overwrite : bool = True
    __hdf5_data : dict = field(default_factory=dict)
    __hdf5_attributes : dict = field(default_factory=dict)
    
    def __del__(self):
        if isinstance(self.hdf5_file, File):
            self.hdf5_file.close()    
            
    def __setitem__(self, key, item, attr=False):
        if attr:
            self.__hdf5_attributes[key] = item
        self.__hdf5_data[key] = item

    def __getitem__(self, key, attr=False):
        if attr:
            return self.__hdf5_attributes[key]
        return self.__hdf5_data[key]
    
    def keys(self):
        #if attr:
        #    return self.__hdf5_attributes.keys()
        return self.__hdf5_data.keys()
    
    def items(self,attr=False):
        if attr:
            return self.__hdf5_attributes.items()
        return self.__hdf5_data.items()
    
    def create_from_dictionary(self, data : dict, attributes : dict = {}) -> None:
        """Create a data object using a dictionary"""
        self.__hdf5_data = {k:v for k, v in data.item()}
        self.__hdf5_attributes  = {k:v for k, v in attributes.item()}
        
    def read_data_file(self, filename : str) -> None:
        """Implement file reading"""
        
        self.hdf5_file = File(filename,'r')
        
        # Read in all of the data
        self.hdf5_file.visititems(self.hdf5_visitor_function)    
        
    def write_data_file(self, filename : str) -> None:
        """Implements file writing for this data_type"""
        
        if os.path.exists(filename) & self.overwrite:
            os.remove(filename)
        elif os.path.exists(filename) & ~self.overwrite:
            raise FileExistsError(f'{filename} already exists and overwrite={self.overwrite}')
            
        if os.path.exists(filename):
            hdf5_output = File(filename,'a')
        else:
            hdf5_output = File(filename,'w')

        for data_path, data_value in self.__hdf5_data.items():
            if data_path in self.large_datasets: continue
            self.create_groups(hdf5_output, data_path)
            hdf5_output.create_dataset(data_path, data=data_value)

        for attr_path, attr_values in self.__hdf5_attributes.items():
            self.create_groups(hdf5_output, attr_path)
            for attr_name, attr_value in attr_values.items():
                hdf5_output[attr_path].attrs[attr_name] = attr_value

    @staticmethod
    def create_groups(data_file : File, path : str):
        """create hdf5 groups iteratively"""
        
        groups = path.split('/')[:-1] 
        grp = data_file
        for group_name in groups:
            if group_name in grp:
                grp = grp[group_name]
            else:
                grp = grp.create_group(group_name)
                
    def hdf5_visitor_function(self, name : str, node):       
        """Visitor function for reading in HDF5 files"""
        for attr_name, attr_value in node.attrs.items():
            if not name in self.__hdf5_attributes:
                self.__hdf5_attributes[name] = {}
            self.__hdf5_attributes[name][attr_name] = attr_value
            
        if isinstance(node, Dataset):
            if name in self.large_datasets:
                self.__hdf5_data[name] = node
            else:
                self.__hdf5_data[name] = node[...]

PipelineFunction = Callable[[HDF5Data], HDF5Data]

class RepointEdges:
    """                                                                                                                                                         
    Scan Edge Split - Each time the telescope stops to repoint this is defined as the edge of a scan                                                            
    """

    def __call__(self, data, source=''):
        """                                                                                                                                                    
        Expects a level 1 data structure                                                                                                                       
        """
        
        return self.getScanPositions(data)

    @staticmethod
    def get_scan_positions(data : HDF5Data):
        """                                                                                                                                                     
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,                                                   
        provides indices for the positions of scans in masked array                                                                                             
                                                                                                                                                                
        Notes:                                                                                                                                                  
        - We may need to check for vane position too                                                                                                            
        - Iteratively finding the best current fraction may also be needed                                                                                      
        """
        features = data.features 
        scan_status = d['hk/antenna0/deTracker/lissajous_status'][...]
        scan_utc    = d['hk/antenna0/deTracker/utc'][...]
        scan_status_interp = interp1d(scan_utc,scan_status,kind='previous',bounds_error=False,
                                      fill_value='extrapolate')(d['spectrometer/MJD'][...])

        scans = np.where((scan_status_interp == self.scan_status_code))[0]
        diff_scans = np.diff(scans)
        edges = scans[np.concatenate(([0],np.where((diff_scans > 1))[0], [scans.size-1]))]
        scan_edges = np.array([edges[:-1],edges[1:]]).T

        return scan_edges

@dataclass 
class COMAPLevel1(HDF5Data):
    """Some helper functions for Level 1 COMAP Data handling are included"""
    
    vane_bit_flag : int = 13
    
    OBSID_MINIMUM : int = 7_000
    OBSID_MAXIMUM : int = 1_000_000 
    
    VANE_HOT_TEMP_OFFSET : float = 273.15 # K

    @property
    def vane_flag(self):
        """Gets the vane flag using the features register"""
        
        features = self['spectrometer/features']
        non_zero_features = (features != 0)
        features[non_zero_features]  = np.floor(np.log(features[non_zero_features])/np.log(2)).astype(int)
        vane_flags = (features == self.vane_bit_flag)
        
        return vane_flags
    
    @property
    def vane_temperature(self):
        """Get the vane temperature"""
        
        return np.nanmean(self['hk/antenna0/vane/Tvane'][:])/100. + self.VANE_HOT_TEMP_OFFSET
    
    @property
    def tod_shape(self):
        """Get the shape of the spectrometer data"""
        return self['spectrometer/tod'].shape
    
    @property 
    def scan_edges(self):
        return 
    
    @property
    def ra(self):
        return self['spectrometer/pixel_pointing/pixel_ra'] 
    
    @ra.setter 
    def ra(self, v):
        self['spectrometer/pixel_pointing/pixel_ra'] = v
        
    @property
    def dec(self):
        return self['spectrometer/pixel_pointing/pixel_dec'] 
    
    @dec.setter 
    def dec(self, v):
        self['spectrometer/pixel_pointing/pixel_dec'] = v

    @property
    def az(self):
        return self['spectrometer/pixel_pointing/pixel_az'] 
    
    @az.setter 
    def az(self, v):
        self['spectrometer/pixel_pointing/pixel_az'] = v


    @property
    def el(self):
        return self['spectrometer/pixel_pointing/pixel_el'] 
    
    @el.setter 
    def el(self, v):
        self['spectrometer/pixel_pointing/pixel_el'] = v

    
    def tod_loop(self,feeds=True, bands=True, channels=True):
        """Applies a function to all spectrometer/tod data"""
        n_feeds, n_bands, n_channels, n_samples = self.tod_shape

        iterators = []
        if feeds:
            iterators += [np.vstack([np.arange(n_feeds,dtype=int), self['spectrometer/feeds']]).T]
        if bands:
            iterators += [np.arange(n_bands,dtype=int)]
        if channels:
            iterators += [np.arange(n_bands,dtype=int)]
        
        return itertools.product(*iterators)
        
@dataclass 
class COMAPLevel2(HDF5Data):
    """Some helper functions for Level 2 COMAP Data handling are included"""

    filename : str = 'pipeline_output.hdf5'

    def __post_init__(self):
        """Define the expected structure for the Level 2 File"""
        
        self['vane/system_temperature'] = np.zeros(1)
        self['vane/system_gain'] = np.zeros(1)

        self['astro_cal/calibration_factors'] = np.zeros(1)
        
        self['spectrometer/tod'] = np.zeros(1)
        self['spectrometer/tod_stddev'] = np.zeros(1)
        self['spectrometer/MJD'] = np.zeros(1)
        self['spectrometer/feeds'] = np.zeros(1)
        self['spectrometer/bands'] = np.zeros(1)
        self['spectrometer/MJD'] = np.zeros(1)
        self['spectrometer/features'] = np.zeros(1)

        self['spectrometer/pixel_pointing/pixel_ra'] = np.zeros(1)
        self['spectrometer/pixel_pointing/pixel_dec'] = np.zeros(1)
        self['spectrometer/pixel_pointing/pixel_az'] = np.zeros(1)
        self['spectrometer/pixel_pointing/pixel_el'] = np.zeros(1)
        
    def update(self, pipeline_function : PipelineFunction):
        """Update data"""
        data, attrs = pipeline_function.save_data
        for k,v in data.items():
            self[k] = v
        for k,v in attrs.items():
            self.__hdf5_attributes[k] = v

    @property
    def tod(self):
        return self['spectrometer/tod'] 
    
    @tod.setter 
    def tod(self, v):
        self['spectrometer/tod'] = v

    @property
    def mjd(self):
        return self['spectrometer/MJD'] 
    
    @mjd.setter 
    def mjd(self, v):
        self['spectrometer/MJD'] = v
        
    @property
    def system_temperature(self):
        return self['vane/system_temperature'] 
        
    @system_temperature.setter 
    def system_temperature(self, v):
        self['vane/system_temperature'] = v
        
    @property
    def system_gain(self):
        return self['vane/system_gain'] 
        
    @system_gain.setter 
    def system_gain(self, v):
        self['vane/system_gain'] = v