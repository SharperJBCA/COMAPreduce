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
