#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:52:15 2023

@author: sharper
"""
from os import path
from dataclasses import dataclass, field
from .DataHandling import HDF5Data, COMAPLevel1, COMAPLevel2
import time 
import socket 
import os
import h5py 
from datetime import datetime

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# set up logging to file - see previous section for more details
import logging

def set_logging(logfilename, loglevel='INFO'):
    path = os.path.dirname(logfilename)
    if path == '':
        path = '.'
    basename = os.path.basename(logfilename).split('.')[0]
    if rank == 0:
        os.makedirs(path,exist_ok=True)
    comm.Barrier()
    log_level = getattr(logging, loglevel)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=f'{path}/{basename}_{formatted_time}_{socket.gethostname()}_PID{os.getpid()}_rank{rank:02d}.log',
                        filemode='w')

@dataclass
class PipelineFunction:
    """Template class implementing the minimum functions needed for a pipeline routine"""
    STATE : bool = True 
    level2 : COMAPLevel2 = field(default_factory=COMAPLevel2) 
    write : bool = True 
    figure_directory : str = 'figures' 

    @property  
    def save_data(self):
        """Use full path that will be saved into the HDF5 file
        
        Give full path of dataset for data/attrs: i.e. spectrometer/tod
        """
        data  = {}
        attrs = {}

        return data, attrs

    def bad_data(self):
        """Check if data is bad"""
        return False
    
    def pre_init(self, data : HDF5Data):
        """Function to run before initialising the pipeline"""
        pass

    def __call__(self, data : HDF5Data) -> HDF5Data:
                        
        return self.STATE

    

class Runner:
    """Class for running the pipeline"""
    
    def __init__(self):
        
        self._filelist = []
        self._processes= {}

        self.level2_data = None
        self.level2_data_dir = '.'
        self.level2_prefix = 'Level2_'
        
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

    def is_level2_file(self, filename : str) -> bool:
        """Check if the file is a level 2 file"""
        
        if self.level2_prefix in filename:
            return True
        else:
            return False

    def run_tod(self):
        
        for filename in self._filelist:
            logging.info(f'PROCESSING {path.basename(filename)}')
            time.sleep(rank*15)


            if self.is_level2_file(filename):
                print('Loading level 1 data')
                data = COMAPLevel2(overwrite=False, large_datasets=['spectrometer/tod'])
                self.level2_prefix = 'temp_'
            else:
                print('Loading level 1 data')
                data = COMAPLevel1(overwrite=False, large_datasets=['spectrometer/tod'])
            data.read_data_file(filename)

            self.level2_data = COMAPLevel2(filename=self.data_path(filename, self.level2_prefix))   
            processes = [process(level2=self.level2_data,**kwargs) for (process,kwargs) in self.processes.items()]

            for process in processes:
                logging.info(f'INITIALISING {process.name}')
                process.pre_init(data) 
                print(process.name, not self.level2_data.contains(process), process.overwrite, process.bad_data())
                if (not self.level2_data.contains(process)) | process.overwrite | process.bad_data():
                    logging.info(f'RUNNING {process.name}')
                    print(f'Running process {process.name}')
                    print(process.STATE)
                    if not process(data, self.level2_data): 
                        print('inside break',process.STATE)
                        logging.info(f'{process.name} has stopped processing file') 
                        break 
                    self.level2_data.update(process)
                    if process.write:
                        self.level2_data.write_data_file(f'{self.level2_data_dir}/Level2_{path.basename(filename)}')


    def run_astro_cal(self):
        """Astro cal functions only expect Level 2 file objects""" 
        
        for filename in self._filelist:
            logging.info(f'PROCESSING {path.basename(filename)}')

            processes = [process(**kwargs) for (process,kwargs) in self.processes.items()]

            print('Loading level 2 data')
            data = COMAPLevel2(filename=filename)
            data.read_data_file(filename)
            for process in processes:
                logging.info(f'RUNNING {process.name}')
                if not process(data): 
                    logging.info(f'{process.name} has stopped processing file') 
                    break 
                data.update(process)
                data.write_data_file(f'{filename}')

    def data_path(self, filename : str, level2_prefix : str ) -> str:
        """Get the path for the output level 2 data"""
        
        filename_short = path.basename(filename)
        name = f'{self.level2_data_dir}/{level2_prefix}{filename_short}'
        
        return name