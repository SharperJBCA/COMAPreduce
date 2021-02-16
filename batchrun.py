from mpi4py import MPI
import sys
import numpy as np
from comancpipeline.Tools import ParserClass

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from run_average import main

if __name__ == "__main__":

    
    parameter_fname = sys.argv[1]

    parameters = ParserClass.Parser(parameter_fname)

    filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str)
    #if prefix in sources:
    #    parameters = f'ParameterFiles/inputs_fornax_{prefix}.ini'
    #    classinfo  = f'ParameterFiles/ClassParameters_{prefix}.ini'
    #else:
    #parameters = f'ParameterFiles/inputs_fornax_general.ini'
    classinfo  = f'ParameterFiles/ClassParameters.ini'

    nfiles = len(filelist)
    step = nfiles//size
    start = rank*step
    end = (rank+1)*step
    if end > nfiles:
        end = nfiles

    main(parameter_fname, classinfo, start, end)
