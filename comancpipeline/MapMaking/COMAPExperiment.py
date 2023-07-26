import numpy as np 
from comancpipeline.MapMaking.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root
from comancpipeline.Tools import binFuncs

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def bin_offset_map(pointing,
                   offsets,
                   weights,
                   offset_length,
                   pixel_edges,
                   extend=False):
    """
    """
    if extend:
        z = np.repeat(offsets,offset_length)
    else:
        z = offsets

    m = np.zeros(int(pixel_edges[-1])+1)
    h = np.zeros(int(pixel_edges[-1])+1)
    binFuncs.binValues(m, pointing, weights=z*weights)
    binFuncs.binValues(h, pointing, weights=weights)

    return m, h

class op_Ax:
    def __init__(self,pointing,weights,offset_length,pixel_edges, special_weight=None):
        
        self.pointing = pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges
        self.special_weight=special_weight
        self.sky_map = np.zeros(int(pixel_edges[-1])+1)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1)
        self.tod_out = np.zeros(pointing.size)

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        m, h = bin_offset_map(self.pointing,
                            tod,
                            self.weights,
                            self.offset_length,
                            self.pixel_edges,extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        diff = tod - self.sky_map[self.pointing]


        if not isinstance(self.special_weight,type(None)):
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)
        else:
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

        return sum_diff
    

class op_Ax_offset_binning:
    def __init__(self,pointing,offset_pointing,weights,offset_length,pixel_edges, offset_edges, special_weight=None):
        
        self.pointing = pointing
        self.offset_pointing = offset_pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges
        self.offset_edges = offset_edges
        self.special_weight=special_weight
        self.sky_map = np.zeros(int(pixel_edges[-1])+1)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1)
        self.tod_out = np.zeros(pointing.size)

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        m, h = bin_offset_map(self.pointing,
                            tod,
                            self.weights,
                            self.offset_length,
                            self.pixel_edges,extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        diff = tod - self.sky_map[self.pointing]

        sum_diff, h_diff = bin_offset_map(self.offset_pointing,
                            diff,
                            self.weights,
                            self.offset_length,
                            self.offset_edges,extend=False)

        return sum_diff



def sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    tod_out =_tod-np.repeat(result,offset_length)
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    naive, naive_h = bin_offset_map(_pointing,
                         _tod,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    destriped = sum_map_to_root(destriped)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    
    if rank == 0:
        m  = destriped/destriped_h
        n  = naive/naive_h
        w = destriped_h

        return {'map':m, 'naive':n, 'weights':w,}
    else:
        return {'map':None, 'naive':None, 'weights':None}