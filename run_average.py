from comancpipeline.Analysis.Running import Runner
from comancpipeline.Analysis.VaneCalibration import MeasureSystemTemperature
from comancpipeline.Analysis.Level1Averaging import CheckLevel1File,Level1Averaging, AtmosphereRemoval,Level1AveragingGainCorrection
from comancpipeline.Analysis.Level2Data import AssignLevel1Data, WriteLevel2Data
from comancpipeline.Analysis.AstroCalibration import FitSource
from comancpipeline.Analysis.Statistics import NoiseStatistics, Spikes
from comancpipeline.Analysis.PostCalibration import ApplyCalibration

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np 

def create_tod_processing(filelist_name : str):
    """This is the tod processing loop"""
    filelist =np.loadtxt(filelist_name,dtype=str, ndmin=1)
    
    idx = np.sort(np.mod(np.arange(filelist.size),size)) 
    idx = np.where((idx == rank))[0] 
    
    tod_processing = Runner()
    processes = {
        CheckLevel1File: {'overwrite': True},
        AssignLevel1Data: {'overwrite': False},
        MeasureSystemTemperature: {'overwrite': False},
        AtmosphereRemoval: {'overwrite': False},
        Level1AveragingGainCorrection: {'overwrite': True},
        FitSource: {'overwrite': False, 'calibration': 'TauA'},
        Spikes: {'overwrite': False},
        NoiseStatistics: {'overwrite': False}
    }

    tod_processing.level2_data_dir = '/scratch/nas_comap3/sharper/COMAP/level2_2023/'
    tod_processing.filelist  : List[str]= filelist[idx]
    tod_processing.processes : Dict[str, Dict[str, bool]] = processes

    return tod_processing 

def create_astro_cal(targets, cal_files, source='TauA'):
    """ 
    1) Assign dates to the target files and cal_files
    1a) If check_calibrators:  
    2) Loop through each target file and assign nearest 'good' calibration observation
    3) Apply the nearest calibrator to the target file
    """
    filelist = np.loadtxt(targets,dtype=str, ndmin=1)
    cal_filelist = np.loadtxt(cal_files,dtype=str, ndmin=1)
        
    cal_processing = Runner()
    processes = {ApplyCalibration:{'calibrator_filelist':cal_filelist}}

    cal_processing.level2_data_dir = '/scratch/nas_comap3/sharper/COMAP/level2_2023/'
    cal_processing.filelist  = filelist
    cal_processing.processes = processes

    return cal_processing 

def map_making(targets, cal_files, source='TauA'):
    """ 
    1) Assign dates to the target files and cal_files
    1a) If check_calibrators:  
    2) Loop through each target file and assign nearest 'good' calibration observation
    3) Apply the nearest calibrator to the target file
    """
    filelist = np.loadtxt(targets,dtype=str, ndmin=1)
    cal_filelist = np.loadtxt(cal_files,dtype=str, ndmin=1)
        
    cal_processing = Runner()
    processes = {ApplyCalibration:{'calibrator_filelist':cal_filelist}}

    cal_processing.level2_data_dir = '/scratch/nas_comap3/sharper/COMAP/level2_2023/'
    cal_processing.filelist  = filelist
    cal_processing.processes = processes

    return cal_processing 

def main():
    tod_processing = create_tod_processing('Filelists/fg9.txt')

    tod_processing.run_tod()
    
    #astro_processing = create_astro_cal(targets='Filelists/fg9_level2.txt',
    #                                    cal_files = 'Filelists/TauA_level2.txt',
    #                                    source='TauA') 
    #astro_processing.run_astro_cal() 

if __name__ == "__main__": 
    main()
