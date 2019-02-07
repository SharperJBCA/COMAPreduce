import h5py

class DataStructure(object):

    def __init__(self):
        pass

    def __call__(self,data, filename=''):
        assert isinstance(data.data, h5py._hl.files.File), 'Data is not a h5py file structure'

        self.parsefilename(filename)
        self.run(data)

    def parsefilename(self, filename):
        self.filename = filename

    def run(self,data):
        pass
