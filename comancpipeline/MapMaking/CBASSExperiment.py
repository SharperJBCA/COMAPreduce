# Defines functions for map making that is specific to CBASS
# Changing these functions should be all you need to define any experiment
#
# C-BASS is a correlation polarimeter, so has instantaneous measurements of I, Q, U
# The IQU measurements are rotated to the sky frame via a rotation matrix.

import numpy as np 
from comancpipeline.MapMaking.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root
from comancpipeline.Tools import binFuncs

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


DETECTOR_TO_SKY =  1 
SKY_TO_DETECTOR = -1 

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

    m = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    h = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    binFuncs.binValues_float(m, pointing, weights=z*weights)
    binFuncs.binValues_float(h, pointing, weights=weights)

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
        self.tod_out = np.zeros(pointing.size,dtype=np.float32)
        self.select_I = self.special_weight[2] == 1
        self.select_Q = self.special_weight[2] == 2
        self.select_U = self.special_weight[2] == 3

    def rotate_tod(self, tod, direction=1):

        self.tod_out[self.select_I]  = tod[self.select_I]
        #  Q c + U s = Q_d
        self.tod_out[self.select_Q] = tod[self.select_Q] * self.special_weight[0][self.select_Q] +\
                direction*tod[self.select_U] * self.special_weight[0][self.select_U]
        # -Q s + U c = U_d
        self.tod_out[self.select_U] = direction*tod[self.select_Q] * self.special_weight[1][self.select_Q] +\
                tod[self.select_U] * self.special_weight[1][self.select_U]

        return self.tod_out 

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        if isinstance(self.special_weight,type(None)):
            m_offset, w_offset = bin_offset_map(self.pointing,
                                                tod,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,extend=False)
        else:

            self.rotate_tod(tod, DETECTOR_TO_SKY)
            m,h  = bin_offset_map(self.pointing,
                                                self.tod_out,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,
                                                extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        self.rotate_tod(self.sky_map[self.pointing], SKY_TO_DETECTOR)

        diff = tod - self.tod_out

        #diff = op_Z(self.pointing, 
        #            tod, 
        #            self.sky_map,special_weight=self.special_weight)

        #print(size,rank, np.sum(diff))
        if not isinstance(self.special_weight,type(None)):
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)
        else:
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

        return sum_diff

def sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    destriped_sqrd, destriped_h_sqrd = bin_offset_map(_pointing,
                         tod_out**2,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    tod_out = i_op_Ax.rotate_tod(_tod, DETECTOR_TO_SKY) 
    naive, naive_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    destriped = sum_map_to_root(destriped)
    destriped_sqrd = sum_map_to_root(destriped_sqrd)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    destriped_h_sqrd = sum_map_to_root(destriped_h_sqrd)

    if rank == 0:
        npix = destriped.size//3 
        I  = destriped[:npix]/destriped_h[:npix]
        Q  = destriped[npix:2*npix]/destriped_h[npix:2*npix]
        U  = destriped[2*npix:]/destriped_h[2*npix:]
        Iw = destriped_h[:npix]
        Qw = destriped_h[npix:2*npix]
        Uw = destriped_h[2*npix:]
        I_rms = np.sqrt(destriped_h_sqrd[:npix]/destriped_h[:npix] - I**2)
        Q_rms = np.sqrt(destriped_h_sqrd[npix:2*npix]/destriped_h[npix:2*npix] - Q**2)
        U_rms = np.sqrt(destriped_h_sqrd[2*npix:]/destriped_h[2*npix:] - U**2)

        return {'I':I, 'Q':Q, 'U':U, 'Iw':Iw, 'Qw':Qw, 'Uw':Uw, 'I_rms':I_rms, 'Q_rms':Q_rms, 'U_rms':U_rms}
    else:
        return {'I':None, 'Q':None, 'U':None, 'Iw':None, 'Qw':None, 'Uw':None, 'I_rms':None, 'Q_rms':None, 'U_rms':None}