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
        try:
            self.plot(data)
        except:
            print('Plotting Failed')

    def run(self,data):
        pass

    def plot(self,data):
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


        self.extrasprefix = config.get('Inputs','extras_prefix')

        data_dir = config.get('Inputs','data_dir')
        if data_dir == 'None':
            data_dir = None
        if out_dir == 'None':
            out_dir = None
        if out_extras_dir == 'None':
            out_extras_dir = None

        if isinstance(data_dir, type(None)):
            self.filename = '{}'.format(filename)
        else:
            self.filename = '{}/{}'.format(data_dir,filename)
        self.rank = rank
        self.size = size
        self.mode = mode


        self.out_dir = out_dir # where to store changed data
        self.out_extras_dir = out_extras_dir
        self.extras = {} # contains extra data to write out


        self.dsets= {} # cropped values of input datasets
        self.attrs= {}
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
        if field not in self.dsets:
            self.setdset(field)

        return self.dsets[field]

    def setdset(self, field, desc=None):
        """
        Allocate dataset to memory.
        """

        # First store shape of original data for outputting
        self.ndims[field] = [s for s in self.data[field].shape]

        # Get maximum dimensions of dataset
        slc   = [slice(0,s,None) for s in self.ndims[field]]
        ndims = [s for s in self.ndims[field]]

        # is this field only having a single axis being selected?
        if field in self.selectFields:
            selectAxis  = self.selectFields[field][0]
            selectIndex = self.selectFields[field][1]
            slc[selectAxis]   = slice(selectIndex,
                                      selectIndex + 1)
            ndims[selectAxis] = 1
        elif not isinstance(desc, type(None)) and not isinstance(self.selectType, type(None)):
            # assume that all the select indices are the same
            selectIndex = self.selectFields[self.selectFields.keys()[0]][1]
            # calculate the selctAxis
            selectAxis  = np.where((np.array(desc) == self.selectType))[0]
            slc[selectAxis]   = slice(selectIndex,
                                      selectIndex + 1)
            ndims[selectAxis] = 1

        slcin = [s for s in slc]
        # if this field being split for MPI?
        if field in self.splitFields:
            splitAxis = self.splitFields[field] # this will return an integer index
            self.lo[field], self.hi[field] = self.getDataRange(self.ndims[field][splitAxis])
            slc[splitAxis]   = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
            ndims[splitAxis] = self.hi[field]-self.lo[field]
        elif not isinstance(desc, type(None)) and not isinstance(self.splitType, type(None)):
            # calculate the selctAxis
            splitAxis  = np.where((np.array(desc) == self.splitType))[0]
            self.lo[field], self.hi[field]     = self.getDataRange(self.ndims[field][splitAxis])
            slc[splitAxis]   = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
            ndims[splitAxis] = self.hi[field]-self.lo[field]

        maxDim = 0
        maxStep = 100000 # don't read more than 1000000 values at once?
        Adims = 1
        for i, dim in enumerate(ndims):
            if i != maxDim:
                Adims *= dim
        
        self.dsets[field] = np.empty(ndims, dtype=self.data[field].dtype)
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


                # Big files need splitting on sidebands too... Maximum read size is 2Gb
                
                if (field in Types._COMAPDATA_) and (Types._SIDEBANDS_ in Types._COMAPDATA_[field]):
                    sidebandaxis = np.where((np.array(Types._COMAPDATA_[field]) == Types._SIDEBANDS_))[0][0]
                    for j in range(self.data[field].shape[sidebandaxis]):
                        slc[sidebandaxis] = slice(j,j+1)
                        slcin[sidebandaxis] = slice(j,j+1)
                        #print(field, slc, slcin, flush=True)
                        self.data[field].read_direct(self.dsets[field], source_sel=tuple(slc), dest_sel=tuple(slcin))
                        #self.dsets[field][tuple(slcin)] = self.data[field][tuple(slc)]
                else:
                    #self.dsets[field][tuple(slcin)] = self.data[field][tuple(slc)]
                    self.data[field].read_direct(self.dsets[field], source_sel=tuple(slc), dest_sel=tuple(slcin))

        else:
            # Else just read the whole thing at once
            self.data[field].read_direct(self.dsets[field], source_sel=tuple(slc), dest_sel=tuple(slcin))

            #self.dsets[field] = self.data[field][tuple(slc)]

    def resizedset(self,field, d):
        """
        Update a dataset with new values.
        Check to see if the dimensions need to be updated.
        """
        #print(self.ndims[field], d.shape)
        factors = [self.dsets[field].shape[i]/d.shape[i] for i in range(len(d.shape))]
        self.dsets[field] = d

        self.ndims[field] = list(d.shape)

        # If we are splitting/selecting any of these axes then we must remember
        # to resize these to full size...
        if field in self.splitFields:
            splitAxis = self.splitFields[field]
            if not Types._SHAPECHANGES_[self.splitType]:
                self.fullFieldLengths[self.splitType] = int( self.fullFieldLengths[self.splitType] / factors[splitAxis])
            self.ndims[field][splitAxis] = self.fullFieldLengths[self.splitType]
            self.lo[field], self.hi[field] = self.getDataRange(self.fullFieldLengths[self.splitType])
            Types._SHAPECHANGES_[self.splitType] = True

        # Do we want to really update the select size again? When would this be useful? Piecewise writing?
        if field in self.selectFields:
            #if not Types._SHAPECHANGES_[self.selectType]:
            selectAxis = self.selectFields[field][0]
            self.ndims[field][selectAxis] = self.fullFieldLengths[self.selectType]
            Types._SHAPECHANGES_[self.selectType] = True

    def updatedset(self,field, d, fromfile=False):
        """
        Update a dataset with new values.
        Check to see if the dimensions need to be updated.
        """
        if fromfile:
            self.setdset(field)
        else:
            self.dsets[field] = d


    def getAttr(self,grp, attr):
        """
        set an attribute in a group in the output data file(self.output)
        """
        if not hasattr(self, 'output'):
            return None

        if (grp in self.attrs) and (attr in self.attrs[grp]):
            return self.attrs[grp][attr]
        elif (grp in self.data) and (attr in self.data[grp].attrs):
            return self.data[grp].attrs[attr]
        else:
            return None

    def setAttr(self,grp, attr, value):
        """
        set an attribute in a group in the output data file(self.output)
        """
        if not hasattr(self, 'output'):
            return None

        if not grp in self.attrs:
            self.attrs[grp] = {}

        self.attrs[grp][attr] = value

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
            self.data = h5py.File(self.filename, driver='mpio',comm=comm)
            self.data.atomic = True # keep data files in synch on all processes
        else:
            self.data = h5py.File(self.filename)
        
        self.fullFieldLengths = Types.getFieldFullLength([self.splitType, self.selectType], self.data)

        if not isinstance(self.out_dir, type(None)):
            if comm.size > 1:
                self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a',
                                        driver='mpio', comm=comm)
                self.output.atomic = True # keep data files in synch on all processes
            else:
                self.output = h5py.File(self.out_dir+'/'+self.data.filename.split('/')[-1],'a')

        if not isinstance(self.out_extras_dir, type(None)):
            extrasname = self.data.filename.split('/')[-1].split('.h')[0]
            if comm.size > 1:
                self.outputextras = h5py.File(self.out_extras_dir+'/'+extrasname+'_{}.hd5'.format(self.extrasprefix),'a',
                                              driver='mpio', comm=comm)#**self.filekwargs)
                self.outputextras.atomic = True # keep data files in synch on all processes
            else:
                self.outputextras = h5py.File(self.out_extras_dir+'/'+extrasname+'_{}.hd5'.format(self.extrasprefix),'a')

    def getextra(self, field, desc=None):
        if field not in self.extras:
            return None

        return self.extras[field][0]

    def resizeextra(self, d, desc):
        
        selectAxis = np.where((np.array(desc) == self.selectType))[0]
        splitAxis  = np.where((np.array(desc) == self.splitType ))[0]

        if (len(selectAxis) == 0) and (len(splitAxis) == 0):
            return d
        else:
            slc = [slice(0,s) for s in d.shape]
            if not (len(selectAxis)==0):
                slc[selectAxis[0]] = slice(self.selectIndex, self.selectIndex+1)
            if not (len(splitAxis)==0):
                splitFull = self.fullFieldLengths[self.splitType]
                lo, hi    = self.getDataRange(splitFull)
                slc[splitAxis[0]] = slice(lo, hi)

            return d[tuple(slc)]

    def setextra(self, key, data, desc):
        """
        splitAxes = [True, True, True, N]
        where the True describes the axis that is split
        """
        
        a = np.where((np.array(desc) == self.selectType))[0]
        if len(a) == 0:
            a = None
        else:
            a = a[0]
        b = np.where((np.array(desc) == self.splitType))[0]
        if len(b) == 0:
            b = None
        else:
            b = b[0]

        self.extras[key] = [data, desc, a , b]



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

    def outputDsets(self, field):
        """
        Output datasets to output file
        """

        # Get maximum dimensions of dataset
        slc   = [slice(0,s,None) for s in self.ndims[field]]
        slcin = [s for s in slc]
        ndims = [s for s in self.ndims[field]]

        if field in self.splitFields:
            splitAxis = self.splitFields[field]
            slc[splitAxis]   = slice(self.lo[field], self.hi[field])
            slcin[splitAxis] = slice(0, self.hi[field]-self.lo[field])
            ndims[splitAxis] = self.fullFieldLengths[self.splitType]

        if field in self.selectFields:
            selectAxis  = self.selectFields[field][0]
            selectIndex = self.selectFields[field][1]
            slc[selectAxis]   = slice(0,1) #slice(selectIndex, selectIndex + 1)
            slcin[selectAxis] = slice(0,1)
            ndims[selectAxis] = 1


        # All mpi nodes must create the same overall sized dataset

        if type(self.dsets[field].flatten()[0]) == np.bytes_:
            tmp = self.output.create_dataset(field, ndims, dtype='S10')
        else:
            tmp = self.output.create_dataset(field, ndims, dtype=self.dsets[field].dtype)

        if field in self.splitFields: # If  in split fields each process has a different dataset
            tmp[tuple(slc)] = self.dsets[field][tuple(slcin)]
        # If it is not in split fields then all data is the same on each process
        # therefore only write out from rank == 0.
        if not (field in self.splitFields) and (comm.rank == 0):
            tmp[tuple(slc)] = self.dsets[field][tuple(slcin)]



    def outputAttrs(self, grp, attr):
        if not grp in self.output:
            self.output.create_group(grp)

        self.output[grp].attrs[attr] = self.attrs[grp][attr]
                    

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
        #synch(10000)


# MPI FUNCTIONS FOR WRITING OUT
def writeoutput(h5data):

    keys = list(h5data.dsets.keys())
    if len(keys) == 0:
        return None

    keys = np.sort(np.array(keys)) # all nodes must do this in the same order
    for k in keys:
        h5data.outputDsets(k)
        #synch(20000)


    # Output any new attributes too
    grps = list(h5data.attrs.keys()) # grps
    if len(grps) == 0:
        return None

    for g in grps:
        attrs = np.sort(np.array(list(h5data.attrs[g].keys())))
        for attr in attrs:
            h5data.outputAttrs(g,attr)
