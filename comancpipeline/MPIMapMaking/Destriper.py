"""
Destriper.py -- An MPI ready implementation of the Destriping algorithm.

Includes a test script + some methods simulating noise and signal

run Destriper.test() to run example script.

Requires a wrapper that will creating the pointing, weights and tod vectors
that are needed to pass to the Destriper.

This implementation does not care about the coordinate system

Refs:
Sutton et al. 2011 

"""
import numpy as np
from matplotlib import pyplot
from scipy.sparse.linalg import LinearOperator
from comancpipeline.Tools import binFuncs
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def share_residual(matvec,x,b):

    local = np.array([(np.linalg.norm(matvec(x) - b))**2])
    if rank == 0:
        allvals = np.zeros(size)
    else:
        allvals = None
    comm.Gather(local,allvals,root=0)
    if rank == 0:
        all_resid = np.array([np.sum(allvals)])
    else:
        all_resid = np.zeros(1)
    comm.Bcast(all_resid,root=0)
    return np.sqrt(np.sum(all_resid))

def share_b(b):

    local = np.array([(np.linalg.norm(b))**2])
    if rank == 0:
        allvals = np.zeros(size)
    else:
        allvals = None
    comm.Gather(local,allvals,root=0)
    if rank == 0:
        all_resid = np.array([np.sum(allvals)])
    else:
        all_resid = np.zeros(1)
    comm.Bcast(all_resid,root=0)
    return np.sqrt(np.sum(all_resid))

def mpi_sum(x):

    local = np.array([np.sum(x)])
    if rank == 0:
        allvals = np.zeros(size)
    else:
        allvals = None
    comm.Gather(local,allvals,root=0)
    if rank == 0:
        all_resid = np.array([np.sum(allvals)])
    else:
        all_resid = np.zeros(1)
    comm.Bcast(all_resid,root=0)
    return np.sum(all_resid)



def cgm(A,b,x0 = None,niter=100,threshold=1e-5,verbose=False):
    """
    Biconjugate CGM implementation from Numerical Recipes 2nd, pg 83-85
    
    arguments:
    b - Array
    Ax - Function that applies the matrix A
    
    kwargs:
    
    Notes:
    1) Ax can be a class with a __call__ function defined to act like a function
    2) Weights should be average weight for an offset, e.g. w_b = sum(w)/offset_length
    3) Need to add a preconditionor matrix step
    """
    
    if isinstance(x0,type(None)):
        x0 = np.zeros(b.size)
    
    r  = b - A.matvec(x0)
    rb = b - A.matvec(x0)
    p  = r*1.
    pb = rb*1.
    
    thresh0 = mpi_sum(r*rb) #np.sum(r*rb)
    for i in range(niter):
        
        q = A.matvec(pb)


        rTrb = mpi_sum(r*rb) #np.sum(r*rb)
        alpha= rTrb/mpi_sum(pb*q)#np.sum(pb * q)

        x0 += alpha*pb
        r = r - alpha*A.matvec(p)
        rb= rb- alpha*A.matvec(pb)
        
        beta = mpi_sum(r*rb)/rTrb
        #np.sum(r*rb)/rTrb
        
        p = r + beta*p
        pb= rb+ beta*pb
        
        delta = mpi_sum(r*rb)/thresh0
        if verbose:
            print(delta)
        if rank ==0:
            print(delta, threshold)
        if delta < threshold:
            break
        
    if (i == (niter-1)):
        print('Convergence not achieved in {} steps'.format(niter))

    print('Final covergence: {} in {:d} steps'.format(delta,i))

    return x0


def bin_offset_map(pointing,
                   offsets,
                   weights,
                   offset_length,
                   pixel_edges,extend=False):
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
    #m = np.histogram(pointing,pixel_edges,
    #                 weights=z*weights)[0]
    #h = np.histogram(pointing,pixel_edges,weights=weights)[0]

    return m, h

def share_map(m,w):

    if rank == 0:
        sum_map = np.zeros((size, m.size))
        wei_map = np.zeros((size, m.size))
    else:
        sum_map = None
        wei_map = None
    comm.Gather(m, sum_map, root=0)
    comm.Gather(w, wei_map, root=0)

    if rank == 0:
        mout = np.sum(sum_map,axis=0)
        wout = np.sum(wei_map,axis=0)
        mout[wout != 0] /= wout[wout != 0]
    else:
        mout = np.zeros(m.size)
        wout = np.zeros(m.size)

    comm.Bcast(mout,root=0)
    comm.Bcast(wout,root=0)

    return mout, wout

def op_Z(pointing, tod, m):
    """
    """
    
    return tod - m[pointing]
    


class op_Ax:
    def __init__(self,pointing,weights,offset_length,pixel_edges):
        
        self.pointing = pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod
        #pyplot.plot(tod)
        #pyplot.show()

        m_offset, w_offset = bin_offset_map(self.pointing,
                                            tod,
                                            self.weights,
                                            self.offset_length,
                                            self.pixel_edges,extend=False)

        m_offset, w_offset = share_map(m_offset,w_offset)

        diff = op_Z(self.pointing, 
                    tod, 
                    m_offset)

        #print(size,rank, np.sum(diff))


        sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

    
        return sum_diff

def run(pointing,tod,weights,offset_length,pixel_edges):

    i_op_Ax = op_Ax(pointing,weights,offset_length,pixel_edges)

    b = i_op_Ax(tod,extend=False)

    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax)

    if rank == 0:
        print('Starting CG')
    x = cgm(A,b)
    if rank == 0:
        print('Done')
    return x

def get_noise(N,sr):

    w = np.random.normal(scale=1,size=N)
    wf = np.fft.fft(w)
    
    u = np.fft.fftfreq(N,d=1./sr)
    u[0] = u[1]
    ps = (np.abs(u)**-2 + 1)

    return np.real(np.fft.ifft(wf*np.sqrt(ps)))

def get_pointing(x,y,npix):
    
    dx = 2./npix
    dy = 2./npix

    xpix = ((x[:]+1)/dx).astype(int)
    ypix = ((y[:]+1)/dy).astype(int)
    pixels = (ypix + xpix*npix).astype(int)

    pixels[(pixels <0) | (pixels >= npix**2)]=-1
    return pixels

def get_signal(N,x,y,npix):
    w = np.random.normal(scale=1,size=(npix,npix))
    wf = np.fft.fft2(w)
    u = np.fft.fftfreq(npix)
    u[0] = u[1]
    u0,u1 = np.meshgrid(u,u)
    r = np.sqrt(u0**2 + u1**2)
    ps = (np.abs(r)**-2 + 1)

    sky = np.real(np.fft.ifft2(wf*np.sqrt(ps)))

    # sky is +/- 1 degrees
    pixels = np.mod(get_pointing(x,y,npix),npix*npix)
    #pixels[pixels > npix*npix] = 0
    sky_tod = sky.flatten()[pixels]
    sky_tod[pixels==-1] = 0
    return sky_tod, pixels, sky

def run_destriper(_pointing,_tod,_weights,offset_length,pixel_edges):
    result = run(_pointing,_tod,_weights,offset_length,pixel_edges)
    m,h = bin_offset_map(_pointing,
                         np.repeat(result,offset_length),
                         _weights,
                         offset_length,
                         pixel_edges,extend=False)

    n,h = bin_offset_map(_pointing,
                         _tod,
                         _weights,
                         offset_length,
                         pixel_edges,extend=False)

    for irank in range(1,size):
        if (rank == 0) & (size > 1):
            m_node2 = np.zeros(m.size)
            print(f'{rank} waiting for {irank}')
            comm.Recv(m_node2, source=irank,tag=irank)
            m += m_node2
            h_node2 = np.zeros(m.size)
            comm.Recv(h_node2, source=irank,tag=irank)
            h += h_node2
            n_node2 = np.zeros(m.size)
            comm.Recv(n_node2, source=irank,tag=irank)
            n += n_node2

        elif (rank !=0 ) & (rank == irank) & (size > 1):
            print(f'{irank} sending to 0')
            comm.Send(m, dest=0,tag=irank)
            comm.Send(h, dest=0,tag=irank)
            comm.Send(n, dest=0,tag=irank)

    #m[h2 != 0] /= h2[h2 != 0]

    #comm.Bcast(h,root=0)

    if rank == 0:
        m[h!=0] /= h[h!=0]
        n[h!=0] /= h[h!=0]

        return {'map':n-m,'naive':n, 'weight':h}
    else:
        return None #{'weight':h}


def test():
    """
    Run an example 1/f + sky signal test case for N threads
    """
    np.random.seed(1)
    T  = 120. # seconds
    sr = 100.
    N = int(T*sr)

    npix = 60
    offset_length = 20
    pixel_edges = np.arange(npix*npix+1).astype(int)

    if rank == 0:
        phase = np.pi/4.
        v = 1./2.
        t = np.arange(N)/sr
        T = N/sr
        
        sky_rate = 1./60.
        offset = T/2.*sky_rate

        x = np.concatenate([t*sky_rate - offset,np.sin(2*np.pi*v*t + phase)])
        y = np.concatenate([np.sin(2*np.pi*v*t + phase),t*sky_rate - offset])

        
        signal,pixels,sky = get_signal(N,x,y,npix)
        noise = get_noise(2*N,sr)


        pixels = get_pointing(x,y,npix)
        tod = noise + signal/3.
        pointing = pixels
        weights = np.ones(tod.size)
        weights[pixels == -1] = 0
        m,h = bin_offset_map(pointing,
                             tod,
                             weights,
                             offset_length,
                             pixel_edges,extend=False)

        m[h != 0] /= h[h != 0]
  

    step = int(2*N/size)
    lo = step*rank
    hi = step*(rank +1)
    if hi > 2*N:
        hi = 2*N

    my_step = hi-lo
    _pointing = np.zeros(my_step,dtype=int)
    _tod = np.zeros(my_step)
    _weights = np.zeros(my_step)
    
    if (rank == 0):
        _pointing[:] = pointing[:my_step]
        _tod[:] = tod[:my_step]
        _weights[:] = weights[:my_step]


    
    for irank in range(1,size):
        if (rank == 0):
            lo = step*irank
            hi = step*(irank +1)
            if hi > 2*N:
                hi = 2*N

            comm.Send(pointing[lo:hi],dest=irank) 
        if (irank == rank) & (rank != 0):
            comm.Recv(_pointing,source=0) 
    for irank in range(1,size):
        if (rank == 0):
            lo = step*irank
            hi = step*(irank +1)
            if hi > 2*N:
                hi = 2*N
            comm.Send(tod[lo:hi],dest=irank) 
        if (irank == rank) & (rank != 0):
            comm.Recv(_tod,source=0) 
    for irank in range(1,size):
        if (rank == 0):
            lo = step*irank
            hi = step*(irank +1)
            if hi > 2*N:
                hi = 2*N
            comm.Send(weights[lo:hi],dest=irank) 
        if (irank == rank) & (rank != 0):
            comm.Recv(_weights,source=0) 

    comm.Barrier()
    maps = run_destriper(_pointing,_tod,_weights,offset_length,pixel_edges)

    if rank == 0:
        pyplot.subplot(211)
        pyplot.imshow(np.reshape(maps['naive'],(npix,npix)))
        pyplot.subplot(212)
        pyplot.imshow(np.reshape(maps['map'],(npix,npix)))
        pyplot.show()


if __name__ == "__main__":
    test()
