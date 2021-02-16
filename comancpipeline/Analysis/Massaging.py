import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Tools import Coordinates, Types
from os import listdir, getcwd
from os.path import isfile, join

#from mpi4py import MPI 
#comm = MPI.COMM_WORLD

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

    def __str__(self):
        if isinstance(self.data_dir, type(None)):
            return "Copying unused fields from data input to data output"
        else:
            return "Copying unused fields to data input from equivalent file in {}".format(self.data_dir)

    def run(self, data):

        # Determine data target and source
        if isinstance(self.data_dir, type(None)):
            cpyTarget = data
            cpySource = data.data
        else:
            cpyTarget = data.data
            filename = cpyTarget.filename.split('/')[-1]
            if comm.size > 1:
                cpySource = h5py.File('{}/{}'.format(self.data_dir, filename), driver='mpio', comm=comm)
                cpySource.atomic = True
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

        def copy_attrs_dict(name,obj):
            if isinstance(obj, h5py.Group):
                if not name in cpyTarget.attrs:
                    cpyTarget.attrs[name] = {}
                #    grp = cpyTarget.create_group(name)
                #else:
                #    grp = cpyTarget[name]

                for key, val in obj.attrs.items():
                    if not key in cpyTarget.attrs[name]:
                        cpyTarget.attrs[name][key] = val
            return None

        if isinstance(cpyTarget, H5Data):
            cpySource.visititems(copy_attrs_dict)
        else:            
            cpySource.visititems(copy_attrs)

        # COPY DATASETS
        def copy_dsets(name,obj):
            if isinstance(obj, h5py.Group):
                if not name in cpyTarget:
                    cpyTarget.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                if not (name in cpyTarget):
                    cpyTarget.create_dataset(name, obj.shape, dtype=obj.dtype)
                    #print(comm.rank, name, obj.shape)
                    if (comm.rank == 0):
                        try:
                            cpyTarget[name][...] = obj[...]
                            #print(obj[...])
                            #print(cpyTarget[name][...])
                        except OSError:
                            # object to big to copy in one, try splitting
                            print('Splitting {} once'.format(name), flush=True)
                            try:
                                for i in range(obj.shape[0]):
                                    cpyTarget[name][i,...] = obj[i,...]
                            except OSError:
                                #STILL TOO BIG!
                                print('Splitting {} again'.format(name), flush=True)
                                for i in range(obj.shape[0]):
                                    for j in range(obj.shape[1]):
                                        cpyTarget[name][i,j,...] = obj[i,j,...]

                    
            return None


        def copy_dsets_dict(name,obj):
            if isinstance(obj, h5py.Dataset):
                if not (name in cpyTarget.dsets):
                    cpyTarget.setdset(name)

                    
            return None

        if isinstance(cpyTarget, H5Data):
            cpySource.visititems(copy_dsets_dict)
        else:            
            cpySource.visititems(copy_dsets)


        comm.Barrier()
