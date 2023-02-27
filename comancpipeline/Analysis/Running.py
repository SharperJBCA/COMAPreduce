#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:52:15 2023

@author: sharper
"""
from os import path
from dataclasses import dataclass, field
from .DataHandling import HDF5Data, COMAPLevel1, COMAPLevel2
    
@dataclass
class PipelineFunction:
    """Template class implementing the minimum functions needed for a pipeline routine"""
    
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2()) 

    @property  
    def save_data(self):
        """Use full path that will be saved into the HDF5 file
        
        Give full path of dataset for data/attrs: i.e. spectrometer/tod
        """
        data  = {}
        attrs = {}

        return data, attrs

    def __call__(self, data : HDF5Data) -> HDF5Data:
                        
        return data

    

class Runner:
    """Class for running the pipeline"""
    
    def __init__(self):
        
        self._filelist = []
        self._processes= []
    
        self.level2_data = None
        self.level2_data_dir = '.'
        self.level2_prefix = 'Level2'
        
    @property
    def filelist(self):
        return self._filelist 
    
    @filelist.setter
    def filelist(self, filelist : list):
        self._filelist = filelist

    @property
    def processes(self) -> list:
        return self._processes 
    
    @processes.setter
    def processes(self, processes : list):
        self._processes = processes

    def __call__(self):
        
        for filename in self._filelist:
            self.level2_data = COMAPLevel2(filename=self.data_path(filename))
            
            data = COMAPLevel1(overwrite=False)
            data.read_data_file(filename)
            for process in self.processes:
                process(data)
                self.level2_data.update(process)

    def data_path(self, filename : str) -> str:
        """Get the path for the output level 2 data"""
        
        filename_short = path.basename(filename)
        name = f'{self.level2_data_dir}/{self.level2_prefix}_{filename_short}'
        
        return name