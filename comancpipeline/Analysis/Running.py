#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:52:15 2023

@author: sharper
"""
from os import path
from dataclasses import dataclass, field
from .DataHandling import HDF5Data, COMAPLevel1, COMAPLevel2

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up logging to file - see previous section for more details
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=f'logs/log_rank{rank:02d}.log',
                    filemode='w')

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
        self.overwrites= []

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
            logging.info(f'PROCESSING {path.basename(filename)}')

            self.level2_data = COMAPLevel2(filename=self.data_path(filename))
            processes = [process(self.level2_data,overwrite=overwrite) for (process,overwrite) in zip(self.processes,
                                                                                  self.overwrites)]

            print('Loading level 1 data')
            data = COMAPLevel1(overwrite=False, large_datasets=['spectrometer/tod'])
            data.read_data_file(filename)
            for process in processes:
                logging.info(f'RUNNING {process.name}')
                if (not self.level2_data.contains(process)) | process.overwrite:
                    if not process(data, self.level2_data): 
                        logging.info(f'{process.name} has stopped processing file') 
                        break 
                    self.level2_data.update(process)
                    self.level2_data.write_data_file(f'{self.level2_data_dir}/Level2_{path.basename(filename)}')

    def data_path(self, filename : str) -> str:
        """Get the path for the output level 2 data"""
        
        filename_short = path.basename(filename)
        name = f'{self.level2_data_dir}/{self.level2_prefix}_{filename_short}'
        
        return name