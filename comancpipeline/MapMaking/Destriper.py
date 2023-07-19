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
import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import gaussian_filter
from comancpipeline.Tools import binFuncs
import healpy as hp

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

DETECTOR_TO_SKY =  1 
SKY_TO_DETECTOR = -1 

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

# Compute the sum of the array x
def mpi_sum(x):

    # Sum the local values
    local = np.array([np.sum(x)])

    comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local[0]

    # # Initialize the array that will store the results from each process
    # if rank == 0:
    #     allvals = np.zeros(size)
    # else:
    #     allvals = None

    # # Gather the local values from each process into the array allvals
    # comm.Gather(local,allvals,root=0)

    # # Sum the results
    # if rank == 0:
    #     all_resid = np.array([np.sum(allvals)])
    # else:
    #     all_resid = np.zeros(1)

    # # Broadcast the sum to the other processes
    # comm.Bcast(all_resid,root=0)

    # # Return the sum
    # return np.sum(all_resid)

def mpi_sum_vector(x):

    local = x
    if rank == 0:
        allvals = np.zeros((size,x.size))
    else:
        allvals = None
    comm.Gather(local,allvals,root=0)
    if rank == 0:
        all_resid = np.sum(allvals,axis=0)
    else:
        all_resid = np.zeros(x.size)
    comm.Bcast(all_resid,root=0)
    return all_resid


def mpi_share_map(m):
    comm.Barrier() 
    if rank == 0:
        print('SHARE MAP') 
        
    for irank in range(1,size):
        if (rank == 0) & (size > 1):
            m_node2 = np.zeros(m.size)
            print(f'{rank} waiting for {irank}')
            comm.Recv(m_node2, source=irank,tag=irank)
            m += m_node2

        elif (rank !=0 ) & (rank == irank) & (size > 1):
            print(f'{irank} sending to 0')
            comm.Send(m, dest=0,tag=irank)
    return m


def cgm(pointing, pixel_edges, tod, weights, obsids, A,b,x0 = None,niter=100,threshold=1e-3,verbose=False, offset_length=50):
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
    
    #rw = weights.reshape((weights.size//offset_length, offset_length)).sum(axis=1) 
    #rfeeds = feedid.reshape((weights.size//offset_length, offset_length)).mean(axis=1) 
    #robs = obsids.reshape((weights.size//offset_length, offset_length)).mean(axis=1) 

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
            print(delta, threshold,flush=True)
        if delta < threshold:
            break
        

    if rank == 0:
        if (i == (niter-1)):
            print('Convergence not achieved in {} steps'.format(niter),flush=True)

        print('Final covergence: {} in {:d} steps'.format(delta,i),flush=True)

    return x0


def bin_offset_map(pointing,
                   offsets,
                   weights,
                   offset_length,
                   pixel_edges,
                   special_weight=None,
                   extend=False):
    """
    """
    if extend:
        z = np.repeat(offsets,offset_length)
    else:
        z = offsets

    m = np.zeros(int(pixel_edges[-1])+1)
    h = np.zeros(int(pixel_edges[-1])+1)
    if isinstance(special_weight,type(None)):
        binFuncs.binValues(m, pointing, weights=z*weights)
        binFuncs.binValues(h, pointing, weights=weights)
    else:
        iqcus = np.zeros(int(pixel_edges[-1])+1) 
        iqsuc = np.zeros(int(pixel_edges[-1])+1) 
        w_iqcus = np.zeros(int(pixel_edges[-1])+1) 
        w_iqsuc = np.zeros(int(pixel_edges[-1])+1) 

        binFuncs.binValues(iqcus, pointing, weights=z*weights*special_weight[0])
        binFuncs.binValues(iqsuc, pointing, weights=z*weights*special_weight[1])

        binFuncs.binValues(w_iqcus, pointing, weights=weights*special_weight[0]**2)
        binFuncs.binValues(w_iqsuc, pointing, weights=weights*special_weight[1]**2)


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

def op_Z(pointing, tod, m, special_weight=None):
    """
    """
    
    if isinstance(special_weight,type(None)):
        return tod - m[pointing]
    else:
        return tod - m[pointing]*special_weight
    

def sum_map_all_inplace(m): 
    comm.Allreduce(MPI.IN_PLACE,
        [m, MPI.DOUBLE],
        op=MPI.SUM
        )
    return m 
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

def run(pointing,tod,weights,offset_length,pixel_edges, obsids, special_weight=None):


    i_op_Ax = op_Ax(pointing,weights,offset_length,pixel_edges,
                    special_weight=special_weight)

    b = i_op_Ax(tod,extend=False)


    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax)

    if rank == 0:
        print('Starting CG',flush=True)
    if False:
        x = cgm(pointing, pixel_edges, tod, weights, obsids, A,b, offset_length=offset_length,threshold=1e-3)
    else:
        x= np.zeros(b.size)
    if rank == 0:
        print('Done',flush=True)
    return x, i_op_Ax

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


def destriper_iteration(_pointing,
                        _tod,
                        _weights,
                        offset_length,
                        pixel_edges,
                        obsids,
                        special_weight=None):
    if isinstance(special_weight,type(None)):
        special_weight = np.ones(_tod.size)

    result,i_op_Ax = run(_pointing,_tod,_weights,offset_length,pixel_edges,
                 obsids,
                 special_weight=special_weight)
    
    if False:
        cbass_map = hp.read_map('/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/calibrated_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits',field=[0,1,2])
        U = hp.ud_grade(cbass_map[2],512)
        idx = _pointing[0] 
        select= (_pointing == (idx+12*512**2*2)) 
        from matplotlib import pyplot
        top = np.sum(_tod[select]*special_weight[0][select])
        bot = np.sum(special_weight[0][select]**2) 
        print(top/bot, idx, U[idx]) 
        pyplot.plot(_tod[select])
        pyplot.plot(U[idx]*special_weight[0][select])
        pyplot.savefig('test.png') 
        pyplot.close()
        pyplot.plot(special_weight[1][select])
        pyplot.savefig('test_pa.png') 
        pyplot.close()
        pyplot.plot(special_weight[0][select])
        pyplot.plot(np.sin(2*special_weight[1][select]))
        pyplot.savefig('test_angle.png') 

    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    naive, naive_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    destriped = sum_map_to_root(destriped)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    
    if rank == 0:
        npix = destriped.size//3 
        I  = destriped[:npix]/destriped_h[:npix]
        Q  = destriped[npix:2*npix]/destriped_h[npix:2*npix]
        U  = destriped[2*npix:]/destriped_h[2*npix:]
        Iw = destriped_h[:npix]
        Qw = destriped_h[npix:2*npix]
        Uw = destriped_h[2*npix:]

        return {'I':I, 'Q':Q, 'U':U, 'Iw':Iw, 'Qw':Qw, 'Uw':Uw}, result
    else:
        return {'I':None, 'Q':None, 'U':None, 'Iw':None, 'Qw':None, 'Uw':None}, result

def run_destriper(_pointing,
                  _tod,
                  _weights,
                  offset_length,
                  pixel_edges,
                  obsids,
                  chi2_cutoff=100,special_weight=None,healpix=False):

    _maps,result = destriper_iteration(_pointing,
                                      _tod,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      obsids,
                                      special_weight=special_weight)

    # Here we will check Chi^2 contributions of each datum to each pixel.
    # Data that exceeds the chi2_cutoff are weighted to zero 
    # Share maps to all nodes
    #for k in maps.keys():
    #    maps[k] = comm.bcast(maps[k],root=0)

    #residual = (_tod-np.repeat(result,offset_length)-maps['map'][_pointing])
    #chi2 = residual**2*_weights
    #_weights[chi2 > chi2_cutoff] = 0
    #_maps,result = destriper_iteration(_pointing,
    #                                  _tod,
    #                                  _weights,
    #                                  offset_length,
    #                                  pixel_edges,
    #                                  feedid,
    #                                  obsids,
    #                                  special_weight=None)


    #if rank == 0:
    #    print('HELLO',flush=True)

    maps = {'All':_maps}
    offsets = {'All':np.repeat(result,offset_length)}
    # now make each individual feed map
    # ufeeds = np.unique(feedid)
    # for feed in ufeeds:
    #     sel = np.where((feedid == feed))[0]
    #     m,h = bin_offset_map(_pointing[sel],
    #                          residual[sel],
    #                          _weights[sel],
    #                          offset_length,
    #                          pixel_edges,extend=False)
    #     m = mpi_share_map(m)
    #     h = mpi_share_map(h)
    #     if rank == 0:
    #         m[h!=0] /= h[h!=0]

    #         maps[f'Feed{feed:02d}']={'map':m,'weight':h}
    #     else:           
    #         maps[f'Feed{feed:02d}']={'map':None,'weight':None}
    
    # for (low,high) in obsid_cuts:
    #     sel = np.where(((obsids >= low) & (obsids < high)))[0]
    #     m,h = bin_offset_map(_pointing[sel],
    #                          residual[sel],
    #                          _weights[sel],
    #                          offset_length,
    #                          pixel_edges,extend=False)
    #     m = mpi_share_map(m)
    #     h = mpi_share_map(h)
    #     if rank == 0:
    #         m[h!=0] /= h[h!=0]

    #         maps[f'ObsID{low:07d}-{high:07d}']={'map':m,'weight':h}
    #     else:           
    #         maps[f'ObsID{low:07d}-{high:07d}']={'map':None,'weight':None}

    # for feed in ufeeds:
    #     for (low,high) in obsid_cuts:
    #         sel = np.where(((obsids >= low) & (obsids < high) & (feedid == feed)))[0]
    #         m,h = bin_offset_map(_pointing[sel],
    #                              residual[sel],
    #                              _weights[sel],
    #                              offset_length,
    #                              pixel_edges,extend=False)
    #         m = mpi_share_map(m)
    #         h = mpi_share_map(h)
    #         if rank == 0:
    #             m[h!=0] /= h[h!=0]

    #             maps[f'Feed{feed:02d}-ObsID{low:07d}-{high:07d}']={'map':m,'weight':h}
    #         else:           
    #             maps[f'Feed{feed:02d}-ObsID{low:07d}-{high:07d}']={'map':None,'weight':None}

    return maps, offsets 

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
    pixel_edges = np.arange(npix*npix).astype(int)

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

    _tod[_tod.size//2+1000] += 100 # TEST SPIKE REMOVAL

    
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
        pyplot.plot(maps['map2']-maps['map']**2)
        pyplot.show()
        pyplot.subplot(131)
        pyplot.imshow(np.reshape(maps['naive'],(npix,npix)))
        pyplot.subplot(132)
        des_map = maps['naive']-maps['map']
        map_rms = np.sqrt(maps['map2']-maps['map']**2)#maps['map2']-des_map**2)
        pyplot.imshow(np.reshape(map_rms,(npix,npix)))
        pyplot.subplot(133)
        map_rms = np.sqrt(1./maps['weight'])#maps['map2'])#-des_map**2)
        pyplot.imshow(np.reshape(maps['map'],(npix,npix)))
        pyplot.show()

def iqu(m,npix):
    return [m[i*npix:(i+1)*npix] for i in range(3)]

def testpol():
    """
    Run an example 1/f + sky signal test case for N threads
    """
    np.random.seed(1)
    T  = 120. # seconds
    sr = 100.
    N = int(T*sr)

    npix = 60
    offset_length = 20
    pixel_edges = np.arange(3*npix*npix).astype(int)

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

        signalP,pixelsP,skyP = get_signal(N,x,y,npix)
        signalP = np.abs(signalP)
        skyP    = np.abs(skyP)
        
        N = N*2
        chi = np.zeros(N) #np.mod(2*np.pi*10*np.linspace(0,1,N),np.pi)

        signalQ = signalP*np.cos(2*chi)
        signalU = signalP*np.sin(2*chi)



        pixels = np.concatenate([get_pointing(x,y,npix),
                                 get_pointing(x,y,npix)+npix**2,
                                 get_pointing(x,y,npix)+npix**2*2]).astype(int)
        tod = np.concatenate([get_noise(N,sr)*0 + signal/3.,
                              get_noise(N,sr)*0 + signalQ/3.,
                              get_noise(N,sr)*0 + signalU/3.])
        pointing = pixels
        weights = np.ones(tod.size)
        weights[pixels == -1] = 0
        special_weight = np.concatenate((np.ones(chi.size),
                                         np.cos(2*chi),
                                         np.sin(2*chi)))
        m,h = bin_offset_map(pointing,
                             tod,
                             weights,
                             offset_length,
                             pixel_edges,
                             special_weight=special_weight,
                             extend=False)


        
        m[h != 0] /= h[h != 0]
        
        [i,q,u] = iqu(m,npix*npix)

        p = np.sqrt(q**2 + u**2)
        pyplot.subplot(121)
        pyplot.imshow(np.reshape(q,(npix,npix)))
        pyplot.subplot(122)
        pyplot.imshow(np.reshape(u,(npix,npix)))
        pyplot.savefig('input_pol.png')
        pyplot.close()

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

    # _tod[_tod.size//2+1000] += 100 # TEST SPIKE REMOVAL

    
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
    az = np.zeros(_tod.size)
    el = np.zeros(_tod.size)
    feedid = np.zeros(_tod.size)
    obsids = np.zeros(_tod.size)
    obsid_cuts = np.zeros(_tod.size)
    maps = run_destriper(_pointing,_tod,_weights,offset_length,pixel_edges,az,el,feedid,obsids,obsid_cuts)
    if rank == 0:
        pyplot.subplot(221)
        pyplot.imshow(np.reshape(maps['All']['map'][:npix*npix],(npix,npix)))
        pyplot.subplot(222)
        P = np.sqrt(np.reshape(maps['All']['map'][npix*npix:2*npix*npix],(npix,npix))**2+\
            np.reshape(maps['All']['map'][2*npix*npix:3*npix*npix],(npix,npix)))
        qout = np.reshape(maps['All']['map'][npix*npix:2*npix*npix],(npix,npix)) 
        uout = np.reshape(maps['All']['map'][2*npix*npix:3*npix*npix],(npix,npix))
        pyplot.imshow(qout)
        pyplot.title('Q')
        pyplot.subplot(223)
        pyplot.imshow(uout)
        pyplot.title('U')
        pyplot.subplot(224)
        pyplot.imshow(np.reshape(skyP,(npix,npix)))
        pyplot.savefig('output_pol.png')
        pyplot.close()


if __name__ == "__main__":
    testpol()
