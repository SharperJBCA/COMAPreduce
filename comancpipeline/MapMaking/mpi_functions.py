
import numpy as np 

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def sum_map_all_inplace(m): 
    """Sum array elements over all MPI processes"""
    comm.Allreduce(MPI.IN_PLACE,
        [m, MPI.DOUBLE],
        op=MPI.SUM
        )
    return m 

def mpi_sum(x):
    """Sum all sums over all MPI processes"""
    # Sum the local values
    local = np.array([np.sum(x)])
    comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local[0]

def sum_map_to_root(m): 
    m_all = np.zeros_like(m) if rank == 0 else None

    # Use MPI Reduce to sum the arrays and store result on rank 0
    comm.Reduce(
        [m, MPI.DOUBLE],
        [m_all, MPI.DOUBLE],
        op=MPI.SUM,
        root=0
    )

    return m_all 
