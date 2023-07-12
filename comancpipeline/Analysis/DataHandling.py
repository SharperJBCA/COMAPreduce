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
from scipy.interpolate import interp1d
import logging
from astropy.time import Time
from datetime import datetime

from comancpipeline.Tools import Coordinates
import glob

def find_file(x, data_dir):
    search_pattern = f"/{data_dir}/comap-{x:07d}-????-??-??-??????.hd5"
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"No files found for obsid {x:07d}.")
        return None
    else:
        for match in matching_files:
            if os.path.exists(match):
                print(f"Found file: {match}")
                return match
        print(f"No existing files found for obsid {x:07d}.")
        return None
    
@dataclass 
class HDF5Data:
    """General class for reading/writing in HDF5 files to memory using dictionaries"""
    
    name : str = 'HDF5Data' 
    large_datasets : list[str] = field(default_factory=list)
    hdf5_file : File = None
    overwrite : bool = True
    __hdf5_data : dict = field(default_factory=dict)
    __hdf5_attributes : dict = field(default_factory=dict)
    
    def __del__(self):
        if self.hdf5_file is not None:
            # Close the HDF5 file, if it is open.
            self.hdf5_file.close()    
            
    def __setitem__(self, key, item):
        # Set the item in the hdf5 data
        self.__hdf5_data[key] = item

    def __getitem__(self, key):
        # Return the key from the hdf5 data
        return self.__hdf5_data[key]
    
    @property 
    def filename(self):
        """ """ 
        return self.hdf5_file.filename

    def attrs(self, path, attribute_key=None):
        if isinstance(attribute_key, type(None)):
            return self.__hdf5_attributes[path]#[attribute_key] 
        else:
            return self.__hdf5_attributes[path][attribute_key] 
    def set_attrs(self, path, attribute_key, value):
        if not path in self.__hdf5_attributes:
            self.__hdf5_attributes[path] = {}

        self.__hdf5_attributes[path][attribute_key] = value

    def keys(self):
        #if attr:
        #    return self.__hdf5_attributes.keys()
        return self.__hdf5_data.keys()
    
    def items(self,attr=False):
        if attr:
            return self.__hdf5_attributes.items()
        return self.__hdf5_data.items()
    
    @property 
    def groups(self):
        """ """ 
        groups = [g.split('/')[0] for g in self.__hdf5_data.keys()] 
        return np.unique(groups)
    
    def create_from_dictionary(self, data : dict, attributes : dict = {}) -> None:
        """Create a data object using a dictionary"""
        self.__hdf5_data = {k:v for k, v in data.item()}
        self.__hdf5_attributes  = {k:v for k, v in attributes.item()}
        
    def read_data_file(self, filename : str) -> None:
        """Implement file reading"""
        logging.info(f'{self.name}: READING {filename}')

        self.hdf5_file = File(filename,'r')
        
        # Read in all of the data
        self.hdf5_file.visititems(self.hdf5_visitor_function)    
        
    def write_data_file(self, filename : str) -> None:
        """Implements file writing for this data_type"""
        logging.info(f'{self.name}: WRITING {filename}')

        if self.hdf5_file is not None:
            self.hdf5_file.close() # Close the file if it is open 

        # If the file already exists, open it in append mode
        if os.path.exists(filename):
            hdf5_output = File(filename,'a')
        # If the file does not exist, open it in write mode
        else:
            hdf5_output = File(filename,'w')

        # Write data to the file
        for data_path, data_value in self.__hdf5_data.items():
            if data_path in self.large_datasets: continue
            self.create_groups(hdf5_output, data_path)
            if data_path in hdf5_output:
                del hdf5_output[data_path] # Delete the dataset if it already exists
            hdf5_output.create_dataset(data_path, data=data_value)

        # Write attributes to the file
        for attr_path, attr_values in self.__hdf5_attributes.items():
            self.create_groups(hdf5_output, attr_path)
            for attr_name, attr_value in attr_values.items():
                hdf5_output[attr_path].attrs[attr_name] = attr_value

        hdf5_output.close()
        self.hdf5_file = File(filename,'r') 

    @staticmethod
    def create_groups(data_file : File, path : str):
        """create hdf5 groups iteratively"""
        
        # split path by '/' to create array of groups
        groups = path.split('/')[:-1] 
        
        # if this an empty group (i.e. just has attributes) 
        if len(groups) == 0: 
            # if the group doesn't exist in the file
            if not path in data_file:
                # create the group
                data_file.create_group(path) 
            
        # start at the top of the file
        grp = data_file
        # for each group name in the array of groups
        for group_name in groups:
            # if the group already exists
            if group_name in grp:
                # move down to that group
                grp = grp[group_name]
            else:
                # if the group doesn't exist, create it
                grp = grp.create_group(group_name)
                
    def hdf5_visitor_function(self, name : str, node):       
        """Visitor function for reading in HDF5 files"""
        for attr_name, attr_value in node.attrs.items():
            if not name in self.__hdf5_attributes:
                self.__hdf5_attributes[name] = {}
            self.__hdf5_attributes[name][attr_name] = attr_value
            
        if isinstance(node, Dataset):
            if name in self.large_datasets:
                self[name] = node
            else:
                self[name] = node[...]

PipelineFunction = Callable[[HDF5Data], HDF5Data]

class RepointEdges:
    """                                                                                                                                                         
    Scan Edge Split - Each time the telescope stops to repoint this is defined as the edge of a scan                                                            
    """

    @staticmethod
    def get_scan_positions(data : HDF5Data, scan_status_code : int = 1):
        """                                                                                                                                                     
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,                                                   
        provides indices for the positions of scans in masked array                                                                                             
                                                                                                                                                                
        Notes:                                                                                                                                                  
        - We may need to check for vane position too                                                                                                            
        - Iteratively finding the best current fraction may also be needed                                                                                      
        """
        if data.source_name in Coordinates.CalibratorList: 
            scan_edges = RepointEdges.get_scan_positions_calibrator(data, scan_status_code)
        else: 
            scan_edges = RepointEdges.get_scan_positions_source(data, scan_status_code)

        return scan_edges
    
    def get_scan_positions_source(data : HDF5Data, scan_status_code : int = 1):
        """                                                                                                                                                     
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,                                                   
        provides indices for the positions of scans in masked array                                                                                             
                                                                                                                                                                
        Notes:                                                                                                                                                  
        - We may need to check for vane position too                                                                                                            
        - Iteratively finding the best current fraction may also be needed                                                                                      
        """
        scan_status = data['hk/antenna0/deTracker/lissajous_status'][...]
        scan_utc    = data['hk/antenna0/deTracker/utc'][...]
        scan_status_interp = interp1d(scan_utc,scan_status,kind='previous',bounds_error=False,
                                      fill_value='extrapolate')(data['spectrometer/MJD'][...])
        if np.sum(scan_status) == 0:
            # instead use the feature bits as we probably have 1 scan 
            select = np.where((data.features == 9))[0] 
            scan_edges = np.array([select[0],select[-1]]).reshape(1,2)
        else:
            scans = np.where((scan_status_interp == scan_status_code))[0]
            diff_scans = np.diff(scans)
            edges = scans[np.concatenate(([0],np.where((diff_scans > 1))[0], [scans.size-1]))]
            scan_edges = np.array([edges[:-1],edges[1:]]).T

        return scan_edges

    
    @staticmethod
    def get_scan_positions_calibrator(data : HDF5Data, scan_status_code : int = 1):
        """                                                                                                                                                     
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,                                                   
        provides indices for the positions of scans in masked array                                                                                             
                                                                                                                                                                
        Notes:                                                                                                                                                  
        - We may need to check for vane position too                                                                                                            
        - Iteratively finding the best current fraction may also be needed                                                                                      
        """

        # Get scan edges
        edges = np.where(data.on_source)[0] 
        scan_edges = np.array([[int(min(edges))], [int(max(edges))]]).T
        return scan_edges


@dataclass 
class COMAPLevel1(HDF5Data):
    """Some helper functions for Level 1 COMAP Data handling are included"""
    
    name : str = 'COMAPLevel1'
    
    vane_bit_flag : int = 13
    bad_keywords : list = field(default_factory=list)
    OBSID_MINIMUM : int = 7_000
    OBSID_MAXIMUM : int = 1_000_000 
    
    VANE_HOT_TEMP_OFFSET : float = 273.15 # K

    def read_data_file_by_obsid(self, obsid, data_directory : str = './'):
        """Reads in a data file by obsid"""
                   
        filename = find_file(obsid, data_directory)
        if isinstance(filename, type(None)):
            return False 
        self.read_data_file(filename)
        return True 
    
    @property 
    def obsid(self):
        try:
            obsid = int(self.attrs('comap','obsid'))
        except KeyError:
            obsid = -1 

        return obsid

    @property 
    def comment(self):
        try:
            comment = self.attrs('comap','comment')
        except KeyError:
            comment = ''
        return comment

    @property 
    def source_name(self):
        try:
            source_split = self.attrs('comap','source').split(',')
        except KeyError:
            source_split = ['']
        if len(source_split) > 1:
            source = [s for s in source_split if s not in self.bad_keywords]
            if len(source) > 0:
                source = source[0]
            else:
                source = ''
        else:
            source = source_split[0]

        return source

    @property
    def on_source(self):
        """ 13 = vane, 0 = off source, 16 = source stare (which we want to ignore)"""
        return (self.features != 13) & (self.features != 0) & (self.features != 16)

    @property
    def vane_flag(self):
        """Gets the vane flag using the features register"""
        features = self.features
        vane_flags = (features == self.vane_bit_flag)
        return vane_flags
    
    @property
    def vane_temperature(self):
        """Get the vane temperature"""
        
        date = Time(self['spectrometer/MJD'][0],format='mjd').datetime 
        if date < datetime(2022, 2, 1):
            tsys = np.nanmean(self['hk/antenna0/vane/Tvane'][:])/100. + self.VANE_HOT_TEMP_OFFSET
        else:
            tshroud = np.nanmean(self['hk/antenna0/vane/Tshroud'][:])/100. + self.VANE_HOT_TEMP_OFFSET
            tsys = 0.2702*tshroud+ 213 # Fitted from date pre 2022-01-01
        return tsys 
    
    @property
    def tod_shape(self):
        """Get the shape of the spectrometer data"""
        return self['spectrometer/tod'].shape
    
    @property
    def frequency(self):
        return self['spectrometer/frequency']

    @property 
    def scan_edges(self):
        return RepointEdges.get_scan_positions(self)
    
    @property
    def features(self):
        if 'spectrometer/features' in self.keys():
            f = self['spectrometer/features']*1
            good = (f != 0)
            f[good] = np.log(f[good])/np.log(2)
            return f.astype(int)
        else:
            raise KeyError('LEVEL 1 FILE CONTAINS NO: spectrometer/features') 
            
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

    @property 
    def source_name(self):
        source_split = self.attrs('comap','source').split(',')
        if len(source_split) > 1:
            source = [s for s in source_split if s not in self.bad_keywords]
            if len(source) > 0:
                source = source[0]
            else:
                source = ''
        else:
            source = source_split[0]

        return source

    @property 
    def airmass(self):
        A = 1./np.sin(self.el*np.pi/180.)
        return A
    
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

    name : str = 'COMAPLevel2'
    filename : str = 'pipeline_output.hdf5'
    vane_bit_flag : int = 13
    bad_keywords : list = field(default_factory=list)

    def __post_init__(self):
        """Define the expected structure for the Level 2 File"""
        
        if os.path.exists(self.filename):
            self.read_data_file(self.filename)
                
    def contains(self, pipeline_function : PipelineFunction):
        """Check if process data is already in level 2 structure"""
        if all([g in self.groups for g in pipeline_function.groups]):
            return True
        else:
            return False
        
    def update(self, pipeline_function : PipelineFunction):
        """Update data"""
        data, attrs = pipeline_function.save_data
        for k,v in data.items():
            if not isinstance(v, type(None)):
                self[k] = v
            
        for k,v in attrs.items():
            for vk, vvalue in v.items():
                self.set_attrs(k,vk,vvalue) 

    @property 
    def source_name(self):
        source_split = self.attrs('comap','source').split(',')
        if len(source_split) > 1:
            source = [s for s in source_split if s not in self.bad_keywords]
            if len(source) > 0:
                source = source[0]
            else:
                source = ''
        else:
            source = source_split[0]

        return source

    @property
    def vane_flag(self):
        """Gets the vane flag using the features register"""
        
        features = self.features
        vane_flags = (features == self.vane_bit_flag)

        return vane_flags

    @property
    def features(self):
        f = self['spectrometer/features']*1
        good = (f != 0)
        f[good] = np.log(f[good])/np.log(2)
        return f.astype(int)
    
    @property
    def on_source(self):
        return (self.features != 13) & (self.features != 0)

    @property 
    def scan_edges(self):
        if not 'averaged_tod/scan_edges' in self.keys():
            return RepointEdges.get_scan_positions(self)
        else:
            return self['averaged_tod/scan_edges']
    @property
    def tod_shape(self):
        return self['averaged_tod/tod'].shape
    
    @property
    def nbands(self):
        return self.tod_shape[1] 

    @property
    def feeds(self):
        return self['spectrometer/feeds'] 

    @property
    def tod(self):
        return self['averaged_tod/tod'] 

    @tod.setter 
    def tod(self, v):
        self['averaged_tod/tod'] = v

    @property
    def mjd(self):
        return self['spectrometer/MJD'] 
    
    @mjd.setter 
    def mjd(self, v):
        self['spectrometer/MJD'] = v
        
    @property
    def ra(self):
        return self['spectrometer/pixel_pointing/pixel_ra'] 
            
    @property
    def dec(self):
        return self['spectrometer/pixel_pointing/pixel_dec'] 
    
    @property
    def az(self):
        return self['spectrometer/pixel_pointing/pixel_az'] 

    @property
    def el(self):
        return self['spectrometer/pixel_pointing/pixel_el'] 

    @property
    def system_temperature_el(self):
        vane_flag = self.vane_flag
        
        vane_indices = np.nonzero(np.diff(vane_flag))[0] + 1
        vane_indices = vane_indices.reshape((vane_indices.size//2,2))
        n_vanes = vane_indices.shape[0]

        return [np.nanmean(self.el[:,v],axis=1) for v in vane_indices]
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
        
    @property 
    def airmass(self):
        A = 1./np.sin(self.el*np.pi/180.)
        return A
    
    def tod_auto_rms(self, ifeed : int, iband : int):
        """Return the auto rms of the feed/band requested"""
        tod = self['averaged_tod/tod'][ifeed,iband] 
        _tod = tod[tod != 0 ]
        N = _tod.size//2 * 2 
        diff = _tod[:N:2]-_tod[1:N:2] 
        return np.nanstd(diff)/np.sqrt(2)

    def tod_loop(self,feeds=True, bands=True):
        """Applies a function to all spectrometer/tod data"""
        n_feeds, n_bands, n_samples = self.tod_shape

        iterators = []
        if feeds:
            iterators += [np.vstack([np.arange(n_feeds,dtype=int), self['spectrometer/feeds']]).T]
        if bands:
            iterators += [np.arange(n_bands,dtype=int)]
        
        return itertools.product(*iterators)
