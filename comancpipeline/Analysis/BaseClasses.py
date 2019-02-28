import h5py
from mpi4py import MPI
from comancpipeline.Tools import Parser, Types
import numpy as np
import configparser
import time
comm = MPI.COMM_WORLD

class DataStructure(object):

    def __init__(self):
        self.mode = 'a'
        self.fields = None
        pass

    def __call__(self,data):
        assert isinstance(data.data, h5py._hl.files.File), 'Data is not a h5py file structure'
        self.run(data)
        self.plot()

    def run(self,data):
        pass

    def plot(self):
        pass

    def __str__(self):
        return "Unknown COMAP Reduce Module"


class DummyTest(DataStructure):
    
    def __init__(self):
        super().__init__()

    def run(self, h5data):
        print(h5data.data.filename)
        h5data.data.create_dataset('a/b',(10,100,), dtype='f')
        h5data.update('new_test{}.h5'.format(h5data.rank))
        print(h5data.data.filename)
        h5data.data.create_dataset('a/b',(10,100,), dtype='f')


class H5Data(object):
    """
    Main data structure for transporting COMAP TOD between pipeline operations.
    Also keeps track of MPI information and sharing of data between cores.
    """

    def __init__(self, filename, rank, size , config,
                 out_extras_dir=None, 
                 out_dir=None,
                 mode='r'):

        if comm.size > 1:
            self.filekwargs = {'driver':'mpio', 'comm':comm}
        else:
            self.filekwargs = {}

        data_dir = config.get('Inputs','data_dir')
        if data_dir == 'None':
            data_dir = './'
        if out_dir == 'None':
            out_dir = None
        if out_extras_dir == 'None':
            out_extras_dir = None

        self.filename = '{}/{}'.format(data_dir,filename)
        self.rank = rank
        self.size = size
        self.mode = mode


        self.out_dir = out_dir # where to store changed data
        self.out_extras_dir = out_extras_dir
        self.extras = {} # contains extra data to write out


        self.dsets= {} # cropped values of input datasets
        self.ndims= {} # original shapes of input datasets (don't use self.data[field].shape)
        self.hi = {}
        self.lo = {}

        try:
            self.splitType = config.get('Inputs','splitAxis')
            if not isinstance(self.splitType, type(None)):
                self.splitType = getattr(Types, self.splitType)
        except (configparser.NoOptionError, configparser.NoSectionError):
            self.splitType = None

        try:
            self.selectType = config.get('SelectAxis','select')
            self.selectIndex = 0
            if not isinstance(self.selectType, type(None)):
                self.selectType  = getattr(Types, self.selectType)
                self.selectIndex = int(config.get('SelectAxis','index'))
        except (configparser.NoOptionError, configparser.NoSectionError):
            self.selectType  = None
            self.selectIndex = 0

        self.splitFields  = Types.getSplitStructure(self.splitType)
        self.selectFields = Types.getSelectStructure(self.selectType, self.selectIndex)

        #self.fields = {}
        #self.selectAxes={}
        #self.splitAxis ={}
        #self.stop = False

    def __del__(self):
        del self.dsets
        del self.extras
        del self.ndims
        self.ndims  = {}
        self.extras = {}
        self.dsets  = {}
        self.hi = {}
        self.lo = {}

    def getdset(self,field):
        """
        Return dataset from hdf5 file and stores it in memory if not already loaded.
        """
        if field not in self.dsets.keys():
            self.setField(field)

        return self.dsets[field]

    def setdset(self,field, d):
        """
        Update a dataset with new values.
        Check to see if the dimensions need to be updated.
        """
        self.dsets[field] = d
        self.ndims[field] = list(d.shape)

        # If we are splitting/selecting any of these axes then we must remember
        # to resize these to full size...
        if field in self.splitFields:
            splitAxis = self.splitFields[field]
            self.ndims[field][splitAxis] = self.fullFieldLengths[self.splitType]
            self.fullFieldLengths[self.splitType] = self.ndims[field][splitAxis]

        # Do we want to really update the select size again? When would this be useful? Piecewise writing?
        if field in self.selectFields:
            selectAxis = self.selectFields[field][0]
            self.ndims[field][selectAxis] = self.fullFieldLengths[self.selectType]
            self.fullFieldLengths[self.selectType] = self.ndims[field][selectAxis]

    def getextra(self, field):
        if field not in self.extras.keys():
            pass

        return self.extras[field][0]

    def setExtra(self,field):
        pass

    def cropExtra(self, d, desc):
        
        selectAxis = np.where((np.array(desc) == self.selectType))[0]
        splitAxis  = np.where((np.array(desc) == self.splitType ))[0]

        if (len(selectAxis) == 0) and (len(splitAxis) == 0):
            return d
        else:
            slc = [s for s in d.shape]
            if not (len(selectAxis)==0):
                slc[selectAxis[0]] = slice(self.selectIndex, self.selectIndex+1)
            if not (len(splitAxis)==0):
                splitFull = self.fullFieldLengths[self.splitType]
                lo, hi    = self.getDataRange(splitFull)
                slc[splitAxis[0]] = slice(lo, hi)

            return d[tuple(slc)]


    def setField(self, field):
        """
        Allocate dataset to memory.
        """

        # First store shape of original data for outputting
        self.ndims[field] = [s for s in self.data[field].shape]
        
        # Get maximum dimensions of dataset
        slc = [slice(0,s,None) for s in self.ndims[field]]
        ndims = [s for s in self.ndims[field]]

        # is this field only having a single axis being selected?
        if field in self.selectFields.keys():
            slc[self.selectFields[field][0]] = slice(self.selectFields[field][1],
                                                     self.selectFields[field][1] + 1)
            ndims[self.selectFields[field][0]] = 1

        slcin = [s for s in slc]
        # if this field being split for MPI?
        if field in self.splitFields.keys():
            splitAxis = self.splitFields[field] # this will return an integer index
            self.lo[field], self.hi[field] = self.getDataRange(self.ndims[field][splitAxis])
            slc[splitAxis] = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
            ndims[splitAxis] = self.hi[field]-self.lo[field]

        maxDim = 0
        maxStep = 1000000 # don't read more than 1000000 values at once?
        Adims = 1
        for i, dim in enumerate(ndims):
            if i != maxDim:
                Adims *= dim
        
        self.dsets[field] = np.zeros(ndims)
        if Adims*ndims[maxDim] > maxStep:
            nsteps = int(Adims*ndims[maxDim]//maxStep)
            if np.mod(Adims*ndims[maxDim],maxStep) != 0:
                nsteps += 1
            stepSize = int(maxStep//Adims) # in first dimension
            if stepSize == 0:
                stepSize = 1
                nsteps = ndims[maxDim]
            for i in range(nsteps):
                lo = i*stepSize
                hi = (i+1)*stepSize
                if i == nsteps-1:
                    hi = ndims[maxDim]

                if field in self.splitFields:
                    if self.splitFields[field] == maxDim:
                        slc[maxDim]   = slice(self.lo[field]+lo,self.lo[field]+hi)
                else:
                    slc[maxDim] = slice(lo,hi)
                slcin[maxDim] = slice(lo,hi)

                self.dsets[field][tuple(slcin)] = self.data[field][tuple(slc)]
        else:
            # Else just read the whole thing at once
            self.dsets[field] = self.data[field][tuple(slc)]



    def setOutputAttr(self,grp, attr, value):
        """
        set an attribute in a group in the output data file(self.output)
        """
        if not hasattr(self, 'output'):
            return None

        if not grp in self.output:
            self.output.create_group(grp)

        self.output[grp].attrs.create(attr, value)

    def open(self, mode='a'):
        """
        Open the input and output files.
        
        self.data == the input file
        self.output == the output file
        
        if no output directory set then self.output = self.data
        (i.e. the data is modified in place)
        """
        self.mode = 'a'
        if comm.size > 1:
            self.data = h5py.File(self.filename,self.mode, driver='mpio',comm=comm)
        else:
            self.data = h5py.File(self.filename,self.mode)

        self.fullFieldLengths = Types.getFieldFullLength([self.splitType, self.selectType], self.data)

        if not isinstance(self.out_dir, type(None)):
            if comm.size > 1:
                self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a',
                                        driver='mpio', comm=comm)
            else:
                self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a')

        if not isinstance(self.out_extras_dir, type(None)):
            extrasname = self.data.filename.split('/')[-1].split('.h')[0]
            if comm.size > 1:
                self.outputextras = h5py.File(self.out_extras_dir+'/'+extrasname+'_Extras.hd5','a',
                                              driver='mpio', comm=comm)#**self.filekwargs)
            else:
                self.outputextras = h5py.File(self.out_extras_dir+'/'+extrasname+'_Extras.hd5','a')


    def outputExtras(self,k):
        """
        If you want to save any extra ancil output, e.g. images, tables, etc... they are saved here.
        """

        # v[0] - the extra data
        # v[1] - the description of the data (e.g. [_HORNS_,_TIME_] or [_OTHER_,_OTHER_])
        # v[2] - Dimension in which a single index is selected
        # v[3] - Dimension in which indices are split between MPI processes

        v = self.extras[k]

        #='mpio', comm=comm)

        if (isinstance(v[2], type(None)) and isinstance(v[3], type(None))):
            tmp = self.outputextras.create_dataset(k, v[0].shape)
            tmp[...] = v[0]
        else:
            slc   = [slice(0,s,None) for s in v[0].shape]
            ndims = [s for s in v[0].shape]
            
            selectAxis = v[2]
            if not isinstance(selectAxis, type(None)):
                selectFull = self.fullFieldLengths[self.selectType]

            splitAxis = v[3]
            if not isinstance(splitAxis, type(None)):
                splitFull = self.fullFieldLengths[self.splitType]
                lo, hi    = self.getDataRange(splitFull)
                slc[splitAxis] = slice(lo,hi)
                ndims[splitAxis] = splitFull

            slc = tuple(slc)
            tmp = self.outputextras.create_dataset(k, ndims) # create the dataset for the correct size
            tmp[slc] = v[0]

    def outputFields(self, field):
        """
        Output datasets to output file
        """

        # Get maximum dimensions of dataset
        slc   = [slice(0,s,None) for s in self.ndims[field]]
        slcin = [s for s in slc]
        ndims = [s for s in self.ndims[field]]

        if field in self.splitFields:
            splitAxis = self.splitFields[field]
            slc[splitAxis] = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])

        if field in self.selectFields:
            selectAxis = self.selectFields[field][0]
            selectIndex = self.selectFields[field][1]
            slc[selectAxis] = slice(selectIndex, selectIndex + 1)
            slcin[selectAxis] = slice(0,1)


        # All mpi nodes must create the same overall sized dataset
        tmp = self.output.create_dataset(field, ndims)
        tmp[tuple(slc)] = self.dsets[field][tuple(slcin)]



    def setExtrasData(self, key, data, axisdesp):
        """
        splitAxes = [True, True, True, N]
        where the True describes the axis that is split
        """
        
        a = np.where((np.array(axisdesp) == self.selectType))[0]
        if len(a) == 0:
            a = None
        else:
            a = a[0]
        b = np.where((np.array(axisdesp) == self.splitType))[0]
        if len(b) == 0:
            b = None
        else:
            b = b[0]

        self.extras[key] = [data, axisdesp, a , b]

    def close(self):
        self.__del__()
        if hasattr(self, 'outputextras'):
            self.outputextras.close()
        if hasattr(self, 'output'):
            self.output.close()
        if hasattr(self,'data'):
            self.data.close()

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

def synch(tag=0):
    buf = None

    if comm.rank == 0:
        for ip in range(1, comm.size):
            comm.send(buf, dest=ip,tag=tag+ip)
            buf = comm.recv(source=ip,tag=tag+ip)
        for ip in range(1,comm.size):
            comm.send(buf,dest=ip,tag=tag+ip)
    else:
        buf = comm.recv(source=0,tag=tag+comm.rank)
        comm.send(buf,dest=0,tag=tag+comm.rank)
        buf = comm.recv(source=0,tag=tag+comm.rank)
# MPI FUNCTIONS FOR WRITING OUT
def writeoutextras(h5data):

    keys = list(h5data.extras.keys())
    if len(keys) == 0:
        return None

    keys = np.sort(np.array(keys)) # all nodes must do this in the same order
    for k in keys:
        h5data.outputExtras(k)
        synch(10000)


# MPI FUNCTIONS FOR WRITING OUT
def writeoutput(h5data):

    keys = list(h5data.dsets.keys())
    if len(keys) == 0:
        return None

    keys = np.sort(np.array(keys)) # all nodes must do this in the same order
    for k in keys:
        h5data.outputFields(k)
        synch(20000)
