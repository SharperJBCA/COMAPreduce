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
from scipy.ndimage import gaussian_filter
from comancpipeline.Tools import binFuncs
import healpy as hp

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

# Compute the sum of the array x
def mpi_sum(x):

    # Sum the local values
    local = np.array([np.sum(x)])

    # Initialize the array that will store the results from each process
    if rank == 0:
        allvals = np.zeros(size)
    else:
        allvals = None

    # Gather the local values from each process into the array allvals
    comm.Gather(local,allvals,root=0)

    # Sum the results
    if rank == 0:
        all_resid = np.array([np.sum(allvals)])
    else:
        all_resid = np.zeros(1)

    # Broadcast the sum to the other processes
    comm.Bcast(all_resid,root=0)

    # Return the sum
    return np.sum(all_resid)

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


def cgm(pointing, pixel_edges, tod, weights, feedid, obsids, A,b,x0 = None,niter=100,threshold=1e-5,verbose=False, offset_length=50):
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
    binFuncs.binValues(m, pointing, weights=z*weights)
    if isinstance(special_weight,type(None)):
        binFuncs.binValues(h, pointing, weights=weights)
    else:
        binFuncs.binValues(h, pointing, weights=weights*special_weight)

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

def op_Z(pointing, tod, m, special_weight=None):
    """
    """
    
    if isinstance(special_weight,type(None)):
        return tod - m[pointing]
    else:
        return tod - m[pointing]*special_weight
    


class op_Ax:
    def __init__(self,pointing,weights,offset_length,pixel_edges, special_weight=None):
        
        self.pointing = pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges
        self.special_weight=special_weight

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
            m_offset, w_offset = bin_offset_map(self.pointing,
                                                tod/self.special_weight,
                                                self.weights*self.special_weight**2,
                                                self.offset_length,
                                                self.pixel_edges,extend=False)

        m_offset, w_offset = share_map(m_offset,w_offset)

        diff = op_Z(self.pointing, 
                    tod, 
                    m_offset,special_weight=self.special_weight)

        #print(size,rank, np.sum(diff))


        sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

    
        return sum_diff

def run(pointing,tod,weights,offset_length,pixel_edges,feedid, obsids, special_weight=None):

    i_op_Ax = op_Ax(pointing,weights,offset_length,pixel_edges,
                    special_weight=special_weight)

    b = i_op_Ax(tod,extend=False)

    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax)

    if rank == 0:
        print('Starting CG',flush=True)
    x = cgm(pointing, pixel_edges, tod, weights, feedid, obsids, A,b, offset_length=offset_length)
    if rank == 0:
        print('Done',flush=True)
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

def destriper_iteration(_pointing,
                        _tod,
                        _weights,
                        offset_length,
                        pixel_edges,
                        feedid,
                        obsids,
                        special_weight=None):
    result = run(_pointing,_tod,_weights,offset_length,pixel_edges,
                 feedid, obsids,
                 special_weight=None)
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
    d2,h = bin_offset_map(_pointing,
                         (_tod-np.repeat(result,offset_length))**2,
                         _weights,
                         offset_length,
                         pixel_edges,extend=False)

    m = mpi_share_map(m)
    h = mpi_share_map(h)
    n = mpi_share_map(n)
    d2 = mpi_share_map(d2)

    if rank == 0:
        m[h!=0] /= h[h!=0]
        n[h!=0] /= h[h!=0]
        d2[h!=0] /= h[h!=0]

        return {'map':n-m,'naive':n, 'weight':h,'map2':d2}, result
    else:
        return {'map':None,'naive':None,'weight':None,'map2':None}  , result

def run_destriper(_pointing,
                  _tod,
                  _weights,
                  offset_length,
                  pixel_edges,
                  az,
                  el,
                  feedid,
                  obsids,
                  obsid_cuts,
                  chi2_cutoff=100,special_weight=None):

    maps,result = destriper_iteration(_pointing,
                                      _tod,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      feedid,
                                      obsids,
                                      special_weight=None)

    # Here we will check Chi^2 contributions of each datum to each pixel.
    # Data that exceeds the chi2_cutoff are weighted to zero 
    # Share maps to all nodes
    for k in maps.keys():
        maps[k] = comm.bcast(maps[k],root=0)

    residual = (_tod-np.repeat(result,offset_length)-maps['map'][_pointing])
    chi2 = residual**2*_weights
    _weights[chi2 > chi2_cutoff] = 0

    # residual = (_tod-np.repeat(result,offset_length))
    # hpix = hp.ang2pix(1024,(90-el)*np.pi/180., az*np.pi/180.)
    # hedges = np.arange(12*1024**2+1)
    # top = np.histogram(hpix, hedges, weights=residual*_weights)[0]
    # bot = np.histogram(hpix, hedges, weights=_weights)[0]
    # top = mpi_sum_vector(top)
    # bot = mpi_sum_vector(bot)

    # mdl = top/bot
    # gd = np.isfinite(mdl)
    # mdl[~gd] = 0
    # if rank == 0:
    #     hp.write_map('maps/azel_map.fits',mdl,overwrite=True)

    # _tod -= mdl[hpix]

    # residual = (_tod-np.repeat(result,offset_length))

    # d_el = gaussian_filter(np.gradient(el,1),11)
    # gd = (residual != 0)
    # nbins = 90

    # el_min = 

    # theta_edges = np.linspace(np.min(d_el[gd]),np.max(d_el[gd]),nbins+1)
    # theta_mids = (theta_edges[1:]+theta_edges[:-1])/2.
    # top = np.histogram(d_el[gd],theta_edges,weights=residual[gd])[0]
    # bot = np.histogram(d_el[gd],theta_edges)[0]

    # if rank ==0:
    #     pyplot.plot(theta_mids, top/bot)
    #     pyplot.show()
    # top = mpi_sum_vector(top)
    # bot = mpi_sum_vector(bot)

    # mdl = top/bot
    # gd = np.isfinite(mdl)
    # mdl_tod = np.interp(d_el, theta_mids[gd], mdl[gd])
    # _tod -= mdl_tod

    #_maps,result = destriper_iteration(_pointing,_tod,_weights,offset_length,pixel_edges)
    _maps,result = destriper_iteration(_pointing,
                                      _tod,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      feedid,
                                      obsids,
                                      special_weight=None)

    # hpix = hp.ang2pix(1024,(90-el)*np.pi/180., az*np.pi/180.)
    # hedges = np.arange(12*1024**2+1)
    # top = np.histogram(hpix, hedges, weights=residual*_weights)[0]
    # bot = np.histogram(hpix, hedges, weights=_weights)[0]
    # top = mpi_sum_vector(top)
    # bot = mpi_sum_vector(bot)


    # mdl = top/bot
    # gd = np.isfinite(mdl)
    # mdl[~gd] = 0
    # if rank == 0:
    #     hp.write_map('maps/azel_map_it2.fits',mdl,overwrite=True)
    # _tod -= mdl[hpix]


    # m,h = bin_offset_map(_pointing,
    #                      np.repeat(result,offset_length),
    #                      _weights,
    #                      offset_length,
    #                      pixel_edges,extend=False)

    # n,h = bin_offset_map(_pointing,
    #                      _tod,
    #                      _weights,
    #                      offset_length,
    #                      pixel_edges,extend=False)
    # d2,h = bin_offset_map(_pointing,
    #                      (_tod-np.repeat(result,offset_length))**2,
    #                      _weights,
    #                      offset_length,
    #                      pixel_edges,extend=False)

    # m = mpi_share_map(m)
    # h = mpi_share_map(h)
    # n = mpi_share_map(n)
    # d2 = mpi_share_map(d2)

    # if rank == 0:
    #     m[h!=0] /= h[h!=0]
    #     n[h!=0] /= h[h!=0]
    #     d2[h!=0] /= h[h!=0]

    #     _maps = {'map':n-m,'naive':n, 'weight':h,'map2':d2}
    # else:
    #     _maps = {'map':None,'naive':None,'weight':None,'map2':None}


    if rank == 0:
        print('HELLO',flush=True)

    maps = {'All':_maps}
    residual = (_tod-np.repeat(result,offset_length))
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

    return maps

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
        pyplot.show()

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
    maps = run_destriper(_pointing,_tod,_weights,offset_length,pixel_edges)

    if rank == 0:
        pyplot.subplot(221)
        pyplot.imshow(np.reshape(maps['map'][:npix*npix],(npix,npix)))
        pyplot.subplot(222)
        P = np.sqrt(np.reshape(maps['map'][npix*npix:2*npix*npix],(npix,npix))**2+\
            np.reshape(maps['map'][2*npix*npix:3*npix*npix],(npix,npix)))
        pyplot.imshow(P)
        #pyplot.subplot(223)
        #pyplot.imshow()
        pyplot.subplot(224)
        pyplot.imshow(np.reshape(skyP,(npix,npix)))

        pyplot.show()



if __name__ == "__main__":
    testpol()
