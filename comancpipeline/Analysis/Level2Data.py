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
    name : str = 'AssignLevel1Data'
    
    overwrite : bool = False

    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    def __post_init__(self):
        
        self.data = {'spectrometer/MJD':np.empty(1),
                'spectrometer/feeds':np.empty(1),
                'spectrometer/bands':np.empty(1),
                'spectrometer/features':np.empty(1),
                'spectrometer/pixel_pointing/pixel_ra':np.empty(1),
                'spectrometer/pixel_pointing/pixel_dec':np.empty(1),
                'spectrometer/pixel_pointing/pixel_az':np.empty(1),
                'spectrometer/pixel_pointing/pixel_el':np.empty(1)}

        self.groups = list(self.data.keys())
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
            
        attrs = {}

        return self.data, attrs

    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2):
        
        self.data['spectrometer/MJD']   = data['spectrometer/MJD']
        self.data['spectrometer/feeds'] = data['spectrometer/feeds']
        self.data['spectrometer/bands'] = data['spectrometer/bands']
        self.data['spectrometer/features']=data['spectrometer/features']
        self.data['spectrometer/pixel_pointing/pixel_ra'] = data['spectrometer/pixel_pointing/pixel_ra']
        self.data['spectrometer/pixel_pointing/pixel_dec']= data['spectrometer/pixel_pointing/pixel_dec']
        self.data['spectrometer/pixel_pointing/pixel_az'] = data['spectrometer/pixel_pointing/pixel_az']
        self.data['spectrometer/pixel_pointing/pixel_el'] = data['spectrometer/pixel_pointing/pixel_el']
        
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