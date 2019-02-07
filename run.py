#from mpi4py import MPI 
#comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser
import sys
import h5py

import optparse
default_parser=optparse.OptionParser(usage="Usage: %prog [options] filenames")
default_parser.add_option("-F","--filelist",dest='filelist',action="store",type='str',default=None,help="Filelist of files to process.")
default_parser.add_option("-P","--parameters",dest='parameters',action="store",type='str',default=None,help="Parameter file describing classes needed for data reduction.")


class H5Data(object):

    def __init__(self, filename, mode='a'):
        self.data = h5py.File(filename,mode)

def getDataRange(rank, size, N):
    """
    get the high/low range for a loop based
    on the MPI rank/size and loop length
    """
        
    allNodes = np.sort(np.mod(np.arange(N).astype(int), size))
    thisNode = np.where((allNodes == rank))[0]

    hi = int(np.max(thisNode)) + 1
    lo = int(np.min(thisNode))
    return lo, hi

import time

def main(args):
    
    options,pargs = default_parser.parse_args(args)

    selectors = Parser.parse_parameters(options.parameters)
    filelist = np.loadtxt(options.filelist, dtype=str,ndmin=1)
    lo, hi = getDataRange(comm.rank, comm.size, filelist.size)

    if comm.rank == 0:
        tstart = time.time()

    for i in range(lo, hi):
        filename = filelist[i]
        print('Opening: {}'.format(filename.split('/')[-1]))
        try:
            for selector in selectors:
                data = H5Data(filename, mode='r')
                print(selector)
                selector(data, filename)
                data.data.close()

            except AssertionError:
                pass

    if comm.rank == 0:
        print('Run time for Root: {} seconds'.format(time.time()-tstart))

if __name__ == "__main__": main(sys.argv[1:])
