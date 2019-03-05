import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Tools import Coordinates, Types
from os import listdir, getcwd
from os.path import isfile, join

from mpi4py import MPI 
comm = MPI.COMM_WORLD

class CheckDsetSizes(DataStructure):
    """
    Changes fullFieldLength dimension to match dset dimensions if they have changed
    """
    
    def __str__(self):
        return "Checking dataset sizes have not changed"


    def run(self,data):

        for axis, changed in Types._SHAPECHANGES_.items():
            if changed:
                for field in Types._COMAPDATA_.keys():
                    if field in data.dsets:
                        try:
                            axisSelect = np.where((np.array(Types._COMAPDATA_[field]) == axis))[0][0]
                            data.fullFieldLengths[axis] = data.ndims[field][axisSelect]
                            break
                        except IndexError:
                            pass

class CopyDsets(DataStructure):

    def __init__(self, data_dir=None):

        if (data_dir == 'None') or (data_dir == None):
            self.data_dir = None
        else:
            self.data_dir = data_dir

    def run(self, data):

        # Determine data target and source
        if isinstance(self.data_dir, type(None)):
            cpyTarget = data.output
            cpySource = data.data
        else:
            cpyTarget = data.data
            filename = cpyTarget.filename.split('/')[-1]
            if comm.size > 1:
                cpySource = h5py.File('{}/{}'.format(self.data_dir, filename), driver='mpio', comm=comm)
            else:
                cpySource = h5py.File('{}/{}'.format(self.data_dir, filename))

        
        # COPY ATTRIBUTES
        def copy_attrs(name,obj):
            if isinstance(obj, h5py.Group):
                if not name in cpyTarget:
                    grp = cpyTarget.create_group(name)
                else:
                    grp = cpyTarget[name]

                for key, val in obj.attrs.items():
                    if not key in grp.attrs:
                        grp.attrs[key] = val
            return None
        cpySource.visititems(copy_attrs)

        # COPY DATASETS
        def copy_dsets(name,obj):
            if isinstance(obj, h5py.Group):
                if not name in cpyTarget:
                    cpyTarget.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                if not (name in cpyTarget) and (comm.rank == 0):
                    try:
                        cpyTarget.create_dataset(name, data=obj)
                    except OSError:
                        # object to big to copy in one, try splitting
                        print('Splitting {} once'.format(name), flush=True)
                        cpyTarget.create_dataset(name, obj.shape)
                        try:
                            for i in range(obj.shape[0]):
                                cpyTarget[name][i,...] = obj[i,...]
                        except OSError:
                            #STILL TOO BIG!
                            print('Splitting {} again'.format(name), flush=True)
                            for i in range(obj.shape[0]):
                                for j in range(obj.shape[1]):
                                    cpyTarget[name][i,j,...] = obj[i,j,...]
                elif not (name in cpyTarget) and (comm.rank > 0):
                    cpyTarget.create_dataset(name, obj.shape)
            return None

        cpySource.visititems(copy_dsets)
