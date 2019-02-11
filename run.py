from mpi4py import MPI 
comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis.BaseClasses import H5Data
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser
import sys
import h5py

import optparse
default_parser=optparse.OptionParser(usage="Usage: %prog [options] filenames")
default_parser.add_option("-F","--filelist",dest='filelist',action="store",type='str',default=None,help="Filelist of files to process.")
default_parser.add_option("-f","--filename",dest='filename',action="store",type='str',default=None,help="Filename to process.")
default_parser.add_option("-P","--parameters",dest='parameters',action="store",type='str',default=None,help="Parameter file describing classes needed for data reduction.")


import time

def main(args):
    
    # Read in the metadata for this pipeline run
    options,pargs = default_parser.parse_args(args)

    selectors, targets, config = Parser.parse_parameters(options.parameters)
    if isinstance(options.filelist, type(None)):
        filelist = np.array([options.filename])
    else:
        filelist = np.loadtxt(options.filelist, dtype=str,ndmin=1)

    
    # Start main loop over each file
    for i in range(filelist.size):
        if comm.rank == 0:
            start = time.time()

        filename = filelist[i]
        print('Opening: {}'.format(filename.split('/')[-1]))
        try: # If the file fails to open we just continue to the next...

            h5data = H5Data(filename,comm.rank, comm.size, out_dir=config.get('Inputs', 'out_dir'))
            h5data.open()

            try: # Check to see if this file has a comment string, no string then no analysis!
                comment = (h5data.data['comap'].attrs['comment'].lower()).decode('utf-8')
            except KeyError:
                print('No comment, skipping...')
                continue

            # Does the comment contain a target we are wanting to analyse?
            check = [t in comment for t in targets]
            if not any(check):
                print('{} does not contain allowed target'.format(comment))
                continue

            # Loop over pipeline operations
            for selector in selectors:

                # Update stored data storage if required...
                h5data.setFields(selector.fields, config) # which fields does this operator need?
                print(selector)
                selector(h5data)
                try: # Run any plotting functions if there are any...
                    selector.plot() # make any plots
                except AttributeError:
                    pass

                if h5data.stop:
                    break
            h5data.close() # Write updated data out to file
        except IOError:
            pass

        if comm.rank == 0:
            print('Run time {}'.format(time.time()-start))


    # for i in getDataRange(comm.rank, comm.size, filelist.size):
    #     if comm.rank == 0:
    #         start = time.time()
    #     filename = filelist[i]
    #     print('Opening: {}'.format(filename.split('/')[-1]))
    #     try:
    #         h5data = H5Data(filename,comm.rank, comm.size)
    #         for selector in selectors:
    #             print(selector)
    #             h5data.open(selector.mode)
    #             selector(h5data)
    #             h5data.close()
    #     except AssertionError:
    #         pass

    #     if comm.rank == 0:
    #         print('Run time {}'.format(time.time()-start))

if __name__ == "__main__": main(sys.argv[1:])
