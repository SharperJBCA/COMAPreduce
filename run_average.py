from comancpipeline.Analysis.Running import Runner
from comancpipeline.Analysis.VaneCalibration import MeasureSystemTemperature
from comancpipeline.Analysis.Level1Averaging import Level1Averaging 
from comancpipeline.Analysis.Level2Data import AssignLevel1Data, WriteLevel2Data

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main():
    
    filelist = ['/path/to/file/comap-0030435-2022-08-11-091554.hd5']
    pipeline = Runner()
    processes = [MeasureSystemTemperature(pipeline.level2_data),
                 Level1Averaging(pipeline.level2_data),
                 AssignLevel1Data(pipeline.level2_data),
                 WriteLevel2Data(pipeline.level2_data)]


    pipeline.filelist  = filelist
    pipeline.processes = processes
    
    pipeline()

if __name__ == "__main__": 
    main()
