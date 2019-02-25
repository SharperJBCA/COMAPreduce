import h5py
from mpi4py import MPI
from comancpipeline.Tools import Parser, Types
import numpy as np

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

        self.filename = filename
        self.rank = rank
        self.size = size
        self.mode = mode

        if out_dir == 'None':
            out_dir = None
        if out_extras_dir == 'None':
            out_extras_dir = None

        self.out_dir = out_dir # where to store changed data
        self.out_extras_dir = out_extras_dir
        self.extras = {} # contains extra data to write out


        self.dsets= {}
        self.hi = {}
        self.lo = {}

        self.splitType = config.get('Inputs','splitAxis')
        if not isinstance(self.splitType, type(None)):
            self.splitType = getattr(Types, self.splitType)
        self.selectType = config.get('SelectAxis','select')
        self.selectIndex = 0
        if not isinstance(self.selectType, type(None)):
            self.selectType  = getattr(Types, self.selectType)
            self.selectIndex = int(config.get('SelectAxis','index'))
        
        self.splitFields  = Types.getSplitStructure(self.splitType)
        self.selectFields = Types.getSelectStructure(self.selectType, self.selectIndex)
        #self.fields = {}
        #self.selectAxes={}
        #self.splitAxis ={}
        #self.stop = False

    def __del__(self):
        del self.dsets
        self.dsets = {}
        #self.fields = {}
        self.hi = {}
        self.lo = {}
        self.data.close()
        if hasattr(self, 'outputextras'):
            print(self.rank, 'CLOSING EXTRAS')
            self.outputextras.close()
            print(self.rank, 'CLOSED EXTRAS')
        if hasattr(self, 'output'):
            self.output.close()

    def getdset(self,field):
        """
        Return dataset from hdf5 file and stores it in memory if not already loaded.
        """
        if field not in self.dsets.keys():
            self.setField(field)

        return self.dsets[field]

    def getextra(self, field):
        if field not in self.extras.keys():
            self.setExtra(field)

        return self.extras[field][0]

    def setExtra(self,field):
        pass

    def setField(self, field):
        """
        Allocate dataset to memory.
        """

        # Get maximum dimensions of dataset
        slc = [slice(0,s,None) for s in self.data[field].shape]
        ndims = [s for s in self.data[field].shape]

        # is this field only having a single axis being selected?
        if field in self.selectFields.keys():
            slc[self.selectFields[field][0]] = slice(self.selectFields[field][1],
                                                     self.selectFields[field][1] + 1)
            ndims[self.selectFields[field][0]] = 1

        slcin = [s for s in slc]
        # if this field being split for MPI?
        if field in self.splitFields.keys():
            splitAxis = self.splitFields[field] # this will return an integer index
            self.lo[field], self.hi[field] = self.getDataRange(self.data[field].shape[splitAxis])
            slc[splitAxis] = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
            ndims[splitAxis] = self.hi[field]-self.lo[field]

        maxDim = np.argmax(ndims)
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
            stepSize = int(maxStep//Adims) # in maximum dimension
            for i in range(nsteps):
                lo = i*stepSize
                hi = (i+1)*stepSize
                if i == nsteps-1:
                    hi = ndims[maxDim]
                slc[maxDim] = slice(lo,hi)
                slcin[maxDim] = slice(lo,hi)
                self.dsets[field][tuple(slcin)] = self.data[field][tuple(slc)]
        else:
            # Else just read the whole thing at once
            self.dsets[field] = self.data[field][tuple(slc)]

    def outputFields(self):
        """
        Output datasets to output file
        """

        # All mpi nodes must create the same overall sized dataset
        if hasattr(self,'output'):
            for field in self.dsets.keys():
                tmp = self.output.create_dataset(field, self.data[field].shape)
                if field in self.dsets.keys():
                    # Get maximum dimensions of dataset
                    slc = [slice(0,s,None) for s in self.data[field].shape]
                    ndims = [s for s in self.data[field].shape]

                    # is this field only having a single axis being selected?
                    if field in self.selectFields.keys():
                        selectAxis = self.selectFields[field][0]
                        selectIndex = self.selectFields[field][1]
                        slc[selectAxis] = slice(selectIndex,
                                                selectIndex + 1)
                        ndims[selectAxis] = 1
                        mustSelect=True
                    else:
                        mustSelect=False

                    slcin = [s for s in slc]
                    # if this field being split for MPI?
                    if field in self.splitFields.keys():
                        splitAxis = self.splitFields[field] # this will return an integer index
                        self.lo[field], self.hi[field] = self.getDataRange(self.data[field].shape[splitAxis])
                        slc[splitAxis] = slice(self.lo[field], self.hi[field])
                        slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
                        mustSplit = True
                    else:
                        mustSplit = False

                    tmp[tuple(slc)] = self.dsets[field][tuple(slcin)]
                else:
                    # if the field is not in the datasets, then copy if from the original datafile.
                    tmp[slc] = self.data[field]

        if len(self.extras.keys()) > 0:
            self.outputExtras()


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
        self.data = h5py.File(self.filename,self.mode, **self.filekwargs)
                              #driver='mpio', comm=comm)

        if not isinstance(self.out_dir, type(None)):
            self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a', **self.filekwargs)
            #, driver='mpio', comm=comm)

        if not isinstance(self.out_extras_dir, type(None)):
            extrasname = self.data.filename.split('/')[-1].split('.h')[0]
            self.outputextras = h5py.File(self.out_extras_dir+'/'+extrasname+'_Extras.hd5','a', **self.filekwargs)


    def outputExtras(self):
        """
        If you want to save any extra ancil output, e.g. images, tables, etc... they are saved here.
        """
        #='mpio', comm=comm)
        self.fullFieldLengths = Types.getFieldFullLength([self.splitType, self.selectType], self.data)


        for k, v in self.extras.items():
            
            if (isinstance(v[2], type(None)) and isinstance(v[3], type(None))):
                tmp = self.outputextras.create_dataset(k, v[0].shape)
                tmp[...] = v[0]
            else:

                slc   = [slice(0,s,None) for s in v[0].shape]
                ndims = [s for s in v[0].shape]

                selectAxis = v[2]
                if not isinstance(selectAxis, type(None)):
                    selectFull = self.fullFieldLengths[self.selectType]
                    #slc[selectAxis] =  

                splitAxis = v[3]
                if not isinstance(splitAxis, type(None)):
                    splitFull = self.fullFieldLengths[self.splitType]
                    lo, hi    = self.getDataRange(splitFull)
                    slc[splitAxis] = slice(lo,hi)
                    ndims[splitAxis] = splitFull

                slc = tuple(slc)
                tmp = self.outputextras.create_dataset(k, ndims) # create the dataset for the correct size
                try:
                    tmp[slc] = v[0]
                except TypeError:
                    print(slc)
                    print(v[0].shape)
                    print(k)
                    raise TypeError

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
