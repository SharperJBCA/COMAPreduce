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
