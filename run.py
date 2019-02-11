from mpi4py import MPI 
comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser
import sys
import h5py

import optparse
default_parser=optparse.OptionParser(usage="Usage: %prog [options] filenames")
default_parser.add_option("-F","--filelist",dest='filelist',action="store",type='str',default=None,help="Filelist of files to process.")
default_parser.add_option("-f","--filename",dest='filename',action="store",type='str',default=None,help="Filename to process.")
default_parser.add_option("-P","--parameters",dest='parameters',action="store",type='str',default=None,help="Parameter file describing classes needed for data reduction.")


class H5Data(object):

    def __init__(self, filename, rank, size , out_dir=None, mode='r'):
        self.filename = filename
        self.rank = rank
        self.size = size
        self.mode = mode
        if isinstance(out_dir, type(None)):
            self.out_dir = ''
        else:
            self.out_dir = out_dir

        self.dsets= {}
        self.hi = {}
        self.lo = {}
        self.fields = {}
        self.selectAxes={}
        self.splitAxis ={}
        self.stop = False

    def __del__(self):
        del self.dsets
        self.dsets = {}
        self.fields = {}
        self.hi = {}
        self.lo = {}
        self.fields = {}
        self.data.close()
        #self.output.close()

    def setFields(self, fields, config):
        """
        Pre-allocate data splits for this core,
        by reading in files from disk.
        """
        if isinstance(fields, type(None)):
            return None
        
        for field, split in fields.items():
            if not field in self.fields.keys(): # Has this data been read in already?
                if split: # Are we splitting this data for MPI purposes?

                    # We remove dimensions we don't care about...
                    # (e.g. select just horn 1, sideband 1...
                    slc = [slice(0,s,None) for s in self.data[field].shape]

                    self.selectAxes[field], self.splitAxis[field] = Parser.parse_split(config, field)
                    if not isinstance(self.selectAxes[field], type(None)):
                        for i in self.selectAxes[field]:
                            slc[i] = slice(i,i+1,None)

                    self.lo[field], self.hi[field] = self.getDataRange(self.data[field].shape[self.splitAxis[field]])
                    slc[self.splitAxis[field]] = slice(self.lo[field],self.hi[field])
                    self.dsets[field] = self.data[field][tuple(slc)]
                    #np.take(grpSliceObj, range(self.lo[field],self.hi[field]), axis=self.splitAxis[field]-len(self.selectAxes))
                                                #[self.lo[field]:self.hi[field],...]
                else:
                    self.dsets[field] = self.data[field][...]
                self.fields[field] = split

    def outputFields(self):
        """
        Write the data back out to disk
        """
        for field, split in self.fields.items():
            print(field, split)
            tmp = self.output.create_dataset(field, self.dsets[field].shape)
            print(tmp)
            print(tmp.shape)
            if split:

                # We remove dimensions we don't care about...
                # (e.g. select just horn 1, sideband 1...
                #if not isinstance(self.selectAxes[field], type(None)):
                #    for i in self.selectAxes[field]:
                #        tmp = tmp[i]

                self.lo[field], self.hi[field] = self.getDataRange(self.data[field].shape[self.splitAxis[field]])

                slc = [slice(0,s,None) for s in self.dsets[field].shape]
                slc[self.splitAxis[field]] = slice(self.lo[field],self.hi[field])
                tmp[tuple(slc)] = self.dsets[field]
            else:
                if self.rank == 0:
                    tmp[...] = self.dsets[field]

    def setOutputAttr(self,grp, attr, value):
        if not grp in self.output:
            self.output.create_group(grp)

        self.output[grp].attrs.create(attr, value)

    def open(self, mode='a'):
        self.mode = 'a'
        self.data = h5py.File(self.filename,self.mode, driver='mpio', comm=comm)

        # output datafile
        #print('dsreduced/'+self.data.filename.split('/')[-1])
        self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a', driver='mpio', comm=comm)

    def close(self):
        self.outputFields()
        self.__del__()

    def update(self, filename, mode='a'):
        self.close()
        self.filename = filename
        self.open(mode)

    def getDataRange(self, N):
        """
        get the high/low range for a loop based
        on the MPI rank/size and loop length
        """
        
        allNodes = np.sort(np.mod(np.arange(N).astype(int), self.size))
        thisNode = np.where((allNodes == self.rank))[0]

        hi = int(np.max(thisNode)) + 1
        lo = int(np.min(thisNode))
        return lo, hi


def getDataRange(rank, size, N):
    """
    get the high/low range for a loop based
    on the MPI rank/size and loop length
    """
        
    allNodes = np.sort(np.mod(np.arange(N).astype(int), size))
    thisNode = np.where((allNodes == rank))[0]

    hi = int(np.max(thisNode)) + 1
    lo = int(np.min(thisNode))
    return range(lo, hi)

import time

def main(args):
    
    options,pargs = default_parser.parse_args(args)

    selectors, targets, config = Parser.parse_parameters(options.parameters)
    if isinstance(options.filelist, type(None)):
        filelist = np.array([options.filename])
    else:
        filelist = np.loadtxt(options.filelist, dtype=str,ndmin=1)

    
    for i in range(filelist.size):
        if comm.rank == 0:
            start = time.time()

        filename = filelist[i]
        print('Opening: {}'.format(filename.split('/')[-1]))
        try:
            h5data = H5Data(filename,comm.rank, comm.size, out_dir=config.get('Inputs', 'out_dir'))
            print(filename)
            h5data.open()
            try:
                comment = (h5data.data['comap'].attrs['comment'].lower()).decode('utf-8')
            except KeyError:
                print('No comment, skipping...')
                continue

            check = [t in comment for t in targets]
            print(comment, targets)
            if not any(check):
                print('{} does not contain allowed target'.format(comment))
                continue

            for selector in selectors:
                h5data.setFields(selector.fields, config) # which fields does this operator need?
                print(selector)
                selector(h5data)
                try:
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
