from mpi4py import MPI
import sys
import numpy as np
from comancpipeline.Tools import ParserClass

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from run_average import main

sources = ['TauA','CasA','CygA','jupiter']
if __name__ == "__main__":

    
    parameter_fname = sys.argv[1]

    parameters = ParserClass.Parser(parameter_fname)

    filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str)
    classinfo  = parameters['Inputs']['classParameters']

    nfiles = len(filelist)
    step = nfiles//size
    start = rank*step
    end = (rank+1)*step
    if end > nfiles:
        end = nfiles

    main(parameter_fname, classinfo, start, end)
