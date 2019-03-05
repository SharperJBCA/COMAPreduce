from mpi4py import MPI 
comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser
import sys
import h5py
import os

import optparse
default_parser=optparse.OptionParser(usage="Usage: %prog [options] filenames")
default_parser.add_option("-F","--filelist",dest='filelist',action="store",type='str',default=None,help="Filelist of files to process.")
default_parser.add_option("-f","--filename",dest='filename',action="store",type='str',default=None,help="Filename to process.")
default_parser.add_option("-P","--parameters",dest='parameters',action="store",type='str',default=None,help="Parameter file describing classes needed for data reduction.")


import time

def main(args):
    print('Starting process {}'.format(comm.rank),flush=True)

    # Read in the metadata for this pipeline run
    options,pargs = default_parser.parse_args(args)

    selectors, targets, config = Parser.parse_parameters(options.parameters)
    if isinstance(options.filelist, type(None)):
        filelist = np.array([options.filename])
    else:
        filelist = np.loadtxt(options.filelist, dtype=str,ndmin=1)

    
    # Start main loop over each file
    for i in range(filelist.size):
        BaseClasses.synch(0)

        if comm.rank == 0:
            start = time.time()

        filename = filelist[i]
        if comm.rank == 0:
            print('Opening: {}'.format(filename.split('/')[-1]),flush=True)

        h5data = BaseClasses.H5Data(filename,comm.rank, comm.size, config,
                                    out_extras_dir=config.get('Inputs', 'out_extras_dir'), 
                                    out_dir=config.get('Inputs', 'out_dir'))

        if config.getboolean('Inputs', 'overwrite') and (comm.rank == 0):
            try:
                if not isinstance(h5data.out_dir, type(None)):
                    os.remove(h5data.out_dir+'/'+filename)
            except OSError:
                pass
            try:
                if not isinstance(h5data.out_extras_dir, type(None)):
                    os.remove(h5data.out_extras_dir+'/'+filename.split('.h')[0]+'_Extras.hd5')
            except OSError:
                pass

        
        BaseClasses.synch(40000) # Ensure no process tries to open a file before it is deleted...
         

        try: # If the file fails to open we just continue to the next...
            h5data.open()
        except OSError:
            print('Could not open file')
            continue

        try: # Check to see if this file has a comment string, no string then no analysis!
            comment = (h5data.getAttr('comap','comment').lower()).decode('utf-8')
        except KeyError:
            print('No comment, skipping...')
            h5data.close()
            continue

        # Does the comment contain a target we are wanting to analyse?
        check = [t in comment for t in targets]
        if not any(check):
            print('{} does not contain allowed target'.format(comment))
            h5data.close()
            continue

        # Loop over pipeline operations
        writeout = True
        for selector in selectors:

            # Update stored data storage if required...
            if comm.rank == 0:
                print(selector,flush=True)
            #try:
            selector(h5data)
            #except KeyError:
            #    print('Failed to find an expected field in {}'.format(filename))
            #    writeout = False
            #    break
            #print(h5data.dsets.keys(),flush=True)

        print(comm.rank, 'About to write', flush=True)

        if hasattr(h5data, 'outputextras') and writeout:
            BaseClasses.writeoutextras(h5data)
        if hasattr(h5data, 'output') and writeout:
            BaseClasses.writeoutput(h5data)

        h5data.close()
        if comm.rank == 0:
            print('Run time {}'.format(time.time()-start))
    print(comm.rank,'at end of script')

if __name__ == "__main__": 
    main(sys.argv[1:])
