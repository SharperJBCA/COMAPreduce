from comancpipeline.Analysis.Running import Runner
from comancpipeline.Analysis.VaneCalibration import MeasureSystemTemperature
from comancpipeline.Analysis.Level1Averaging import CheckLevel1File,Level1Averaging, AtmosphereRemoval,Level1AveragingGainCorrection
from comancpipeline.Analysis.Level2Data import AssignLevel1Data, WriteLevel2Data
from comancpipeline.Analysis.AstroCalibration import FitSource

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np 

def main():
    
    filelist =np.loadtxt('Filelists/fg9.txt',dtype=str, ndmin=1)
    
    idx = np.sort(np.mod(np.arange(filelist.size),size)) 
    idx = np.where((idx == rank))[0] 
    
    #'/scratch/nas_comap2/sharper/COMAP/data/comap-0017347-2021-01-30-201327.hd5']
    #'/path/to/file/comap-0030435-2022-08-11-091554.hd5']
    pipeline = Runner()
    processes = [CheckLevel1File,
                 AssignLevel1Data,
                 MeasureSystemTemperature,
                 AtmosphereRemoval,
                 Level1AveragingGainCorrection,
                 FitSource]
    overwrites = [True,
                 False,
                 False,
                 False,
                 True,
                 False]

    pipeline.level2_data_dir = '/scratch/nas_comap3/sharper/COMAP/level2_2023/'
    pipeline.filelist  = filelist[idx]
    pipeline.processes = processes
    pipeline.overwrites = overwrites

    pipeline()

if __name__ == "__main__": 
    main()
