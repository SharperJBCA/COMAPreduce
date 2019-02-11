import h5py

class DataStructure(object):

    def __init__(self):
        self.mode = 'a'
        self.fields = None
        pass

    def __call__(self,data):
        assert isinstance(data.data, h5py._hl.files.File), 'Data is not a h5py file structure'
        self.run(data)

    def run(self,data):
        pass

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
