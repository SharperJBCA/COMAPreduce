#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:47:59 2023

@author: sharper
"""
from os import path, makedirs
import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 

from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2
from comancpipeline.Tools.stats import auto_rms

@dataclass 
class AssignLevel1Data(PipelineFunction):
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
        data = {'spectrometer/MJD':self.mjd,
                'spectrometer/feeds':self.feeds,
                'spectrometer/bands':self.bands,
                'spectrometer/features':self.features,
                'spectrometer/pixel_pointing/pixel_ra':self.ra,
                'spectrometer/pixel_pointing/pixel_dec':self.dec,
                'spectrometer/pixel_pointing/pixel_az':self.az,
                'spectrometer/pixel_pointing/pixel_el':self.el}
            
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data) -> HDF5Data:
        
        self.mjd   = data['spectrometer/MJD']
        self.feeds = data['spectrometer/feeds']
        self.bands = data['spectrometer/bands']
        self.features=data['spectrometer/features']
        self.ra = data['spectrometer/pixel_pointing/pixel_ra']
        self.dec= data['spectrometer/pixel_pointing/pixel_dec']
        self.az = data['spectrometer/pixel_pointing/pixel_az']
        self.el = data['spectrometer/pixel_pointing/pixel_el']
        
        return data

@dataclass 
class WriteLevel2Data(PipelineFunction):
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

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