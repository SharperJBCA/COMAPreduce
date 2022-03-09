import numpy as np
from matplotlib import pyplot

from comancpipeline.MapMaking import Types, MapTypes, OffsetTypes #import Types.Offsets, Map, HealpixMap, ProxyHealpixMap
from comancpipeline.Tools import binFuncs
from scipy.sparse import lil_matrix, diags, linalg

def binOffsets(offsets,weights,offsetpixels,pixels,npix=9):
    """
    Add more data to the naive map
    """
    sigwei = np.zeros(npix)
    wei    = np.zeros(npix)
    binFuncs.binValues2Map(sigwei, pixels, offsets*weights, offsetpixels)
    binFuncs.binValues2Map(wei   , pixels, 1*weights      , offsetpixels)
    return sigwei, wei

def pixels_to_P(pixels,npix):
    """
    Convert 1D array of pixels into P matrix
    """
    
    P = lil_matrix((len(pixels),npix))
    for i in range(pixels.size):
        P[i,pixels[i]] = 1
        
    return P

def create_F(ntod,offset_len):
    """
    """
    noffsets = int(np.ceil(ntod/offset_len))
    F = lil_matrix((ntod,noffsets))
    
    j = 0
    fpix = np.zeros(ntod)
    for i in range(ntod):
        if (i > 0) & (np.mod(i,offset_len)==0):
            j += 1
        F[i,j] = 1
        fpix[i]=j
    return F, fpix.astype(int)

class Axfunc:
    
    def __init__(self, weights, offset_pix, map_pix,npix,covariance=None):
        
        self.weights = weights
        self.offset_pix = offset_pix
        self.map_pix = map_pix
        self.npix = npix
        self.output = np.zeros(self.weights.size)
        self.covariance = covariance
        if not isinstance(self.covariance,type(None)):
            self.covariance = 1./np.fft.fft(self.covariance)
    
    def __call__(self,xk):
        sigwei,wei = binOffsets(xk,
                                self.weights,
                                self.offset_pix,
                                self.map_pix,self.npix)

        m = sigwei/wei

        self.output *= 0.
        binFuncs.EstimateResidualSimplePrior(self.output, # output
                                             xk, # Ax - b (strected to TOD)
                                             self.weights, # TOD weights
                                             m, # m[Ax-b]
                                             self.offset_pix, 
                                             self.map_pix)

        if not isinstance(self.covariance,type(None)):
            self.output += np.real(np.fft.ifft(np.fft.fft(xk.flatten())*self.covariance))
                

        return self.output

class Axfunc_slow:
    
    def __init__(self, A):
        
        self.A = A
    
    def __call__(self,xk):

        return self.A.dot(xk[:,None]).flatten()



def Destriper(data,
              niter=100,
              offset=50,
              covariance=None,verbose=False,threshold=-5):
    """
    Destriping routines
    """

    niter = int(niter)

    # NB : Need to change offsets to ensure that each
    # is temporally continuous in the future, for now ignore this.
    offsetLen = offset
    threshold = 10**threshold
    verbose   = verbose
    Noffsets  = data.Nsamples//offsetLen

    # Offsets for storing the outputs
    offsets   = OffsetTypes.Offsets(offsetLen, Noffsets,  data.Nsamples)

    # For storing the offsets on the sky
    offsetMap  = MapTypes.FlatMapType(data.naive.crval, 
                                      data.naive.cdelt, 
                                      data.naive.crpix, 
                                      data.naive.ctype,
                                      nxpix=data.naive.nxpix,
                                      nypix=data.naive.nypix)

    #CGM(data, offsets, offsetMap, niter=niter)

    # Calculate the average weight per offset
    weights = np.histogram(data.offset_residuals.offsetpixels,
                           np.arange(data.offset_residuals.Noffsets+1),
                           weights=data.all_weights)[0]/data.offset_residuals.offset_length

    Ax = Axfunc(weights,
                data.offset_residuals.offsetpixels,
                data.pixels,
                offsetMap.npix,
                covariance=covariance)

    offsets.offsets = CGM(data.offset_residuals.sig, Ax, niter=niter,verbose=verbose,threshold=threshold)

    # Bin the offsets in to a map
    offsetMap.sum_offsets(offsets.offsets,
                          weights,
                          offsets.offsetpixels,
                          data.pixels)
    offsetMap.average()



    return offsetMap, offsets

def DestriperHPX(parameters, data,covariance=None):
    """
    Destriping routines
    """
    niter = int(parameters['Destriper']['niter'])

    # NB : Need to change offsets to ensure that each
    # is temporally continuous in the future, for now ignore this.
    offsetLen = parameters['Destriper']['offset']
    Noffsets  = data.Nsamples//offsetLen

    # Offsets for storing the outputs
    offsets   = Types.Offsets(offsetLen, Noffsets,  data.Nsamples)

    # For storing the offsets on the sky
    offsetMap = Types.ProxyHealpixMap(data.naive.nside,npix=data.naive.npix)
    offsetMap.uni2pix = data.naive.uni2pix



    # Calculate the average weight per offset
    weights = np.histogram(data.residual.offsetpixels,np.arange(data.residual.Noffsets+1),weights=data.allweights)[0]/data.residual.offset

    # Define out Ax linear operator function
    Ax = Axfunc(weights,data.residual.offsetpixels,data.pixels,offsetMap.npix,
                covariance=covariance)

    # Run the CGM code
    offsets.offsets = CGM(data.residual.sigwei, Ax, niter=niter,verbose=True,threshold=1e-7)

    # Bin the offsets in to a map
    offsetMap.binOffsets(offsets.offsets,
                         weights,
                         offsets.offsetpixels,
                         data.pixels)
    offsetMap.average()

    # Variance map
    offsetMap.binOffsets(offsets.offsets,
                         weights,
                         offsets.offsetpixels,
                         data.pixels)
    offsetMap.average()

    return offsetMap, offsets



def CGM(b,Ax,x0 = None,niter=100,threshold=1e-7,verbose=False):
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
    
    r  = b.flatten() - Ax(x0)
    rb = b.flatten() - Ax(x0)
    p  = r*1.
    pb = rb*1.
    
    thresh0 = np.sum(r*rb)
    for i in range(niter):
        
        q = Ax(pb)


        rTrb = np.sum(r*rb)
        alpha= rTrb/np.sum(pb * q)

        x0 += alpha*pb
        r = r - alpha*Ax(p)
        rb= rb- alpha*Ax(pb)
        
        beta = np.sum(r*rb)/rTrb
        
        p = r + beta*p
        pb= rb+ beta*pb
        
        delta = np.sum(r*rb)/thresh0
        if verbose:
            print(delta)
        if delta < threshold:
            break
        
    if (i == (niter-1)):
        print('Convergence not achieved in {} steps'.format(niter))

    print('Final covergence: {} in {:d} steps'.format(delta,i))

    return x0
    

def CGM_old(data, offsets, offsetMap, niter=400,verbose=False):
    """
    Conj. Gradient Inversion
    """

    # -- We are performing inversion of Ax = b    
    # Solving for x, Ax = b
    Ax = Types.Offsets(offsets.offset, offsets.Noffsets, offsets.Nsamples)
    b  = data.residual
    counts = offsets.offsets*0.

    b.average()
    Ax.average()

    # Estimate initial residual
    binFuncs.EstimateResidualSimplePrior(Ax.sigwei, # Holds the weighted residuals
                                         offsets.offsets, # holds the target offsets
                                         data.allweights, # Weights per TOD sample
                                         offsetMap.output, # Map to store the offsets in (initially all zero)
                                         offsets.offsetpixels, # Maps offsets to TOD position
                                         data.pixels)
    
    if verbose:
        print('Diag counts:',np.min(counts))

    # -- Calculate the initial residual and direction vectors
    if verbose:
        print('Diags b.sigwei, Ax.offsets:', np.sum(b.sigwei), np.sum(Ax.offsets))

    residual = b.sigwei - Ax.sigwei
    direction= b.sigwei - Ax.sigwei


    r2 = b.sigwei - Ax.sigwei

    # -- Initial threshhold
    thresh0 = np.sum(residual**2)
    dnew    = np.sum(residual**2)
    alpha   = 0

    if verbose:
        print('Diags thresh0:', thresh0)
    #offsets.offsets = data.residual.offsets 
    lastoffset = 0
    newVals = np.zeros(niter)
    alphas  = np.zeros(niter)
    betas   = np.zeros(niter)
    if np.isnan(np.sum(b.sigwei)):
        return

    # --- CGM loop ---
    for i in range(niter):
        # -- Calculate conjugate search vector Ad
        lastoffset = Ax.sigwei*1.
        Ax.sigwei *= 0
        counts *= 0

        offsetMap.clearmaps()

        # CGM overview:
        # 0) d = Ax - b
        # 1) q = A * d
        # 2) alpha = rTr / dTAd = rTr/dTq

        # We apply the A matrix to direction
        # 1) Stretch out Ax-b to length of TOD
        # 2) Create a weighted average map
        # 3) Find residual between (Ax-b) and P m[Ax-b]
        # 4) Produce a weighted sum into the offsets

        # Here we create the map m[Ax-b]
        offsetMap.binOffsets(direction,
                             data.residual.wei,
                             offsets.offsetpixels,
                             data.pixels)
        offsetMap.average() # And properly weight it

        # Here we find the residuals, and sum into offsets
        binFuncs.EstimateResidualSimplePrior(Ax.sigwei, # output
                                             direction, # Ax - b (strected to TOD)
                                             data.allweights, # TOD weights
                                             offsetMap.output, # m[Ax-b]
                                             offsets.offsetpixels, 
                                             data.pixels)

                         
        

        # Calculate the search vector
        dTq = np.sum(direction*Ax.sigwei)


        # 
        alpha = dnew/dTq
        alphas[i] = alpha
        # -- Update offsets

        olfast = offsets.offsets*1.
        offsets.offsets += alpha*direction


        #offsets.offsets[0] = offsets.offsets[1]

        # -- Calculate new residual
        if False:#np.mod(i,50000) == 0:
            offsetMap.clearmaps()
            offsetMap.binOffsets(offsets.offsets,
                                 data.residual.wei,
                                 offsets.offsetpixels,
                                 data.pixels)
            offsetMap.average()
            Ax.sigwei *= 0
            counts = offsets.offsets*0.

            binFuncs.EstimateResidualSimplePrior(Ax.sigwei, # Holds the weighted residuals
                                                 counts,
                                                 offsets.sigwei, # holds the target offsets
                                                 data.allweights,#residual.wei,
                                                 offsetMap.output, # Map to store the offsets in (initially all zero)
                                                 offsets.offsetpixels, # Maps offsets to TOD position
                                                 data.pixels)

            residual = b.sigwei - Ax.sigwei
        else:
            from matplotlib import pyplot
            pyplot.subplot(2,1,1)
            pyplot.plot(residual)
            residual = residual -  alpha*Ax.sigwei 

        dold = dnew*1.0
        dnew = np.sum(residual**2)
        newVals[i] = dnew
        print(dnew,dold,alpha)
        pyplot.plot(residual)
        pyplot.subplot(2,1,2)
        pyplot.plot(alpha*Ax.sigwei )
        pyplot.show()

        # --
        beta = dnew/dold
        betas[i] = beta

        # -- Update direction
        direction = residual + beta*direction

        offsetMap.clearmaps()
        offsetMap.binOffsets(direction,
                             data.residual.wei,
                             offsets.offsetpixels,
                             data.pixels)
        offsetMap.average()
                   
        

        print((-np.log10(dnew/thresh0))/8 )
        if dnew/thresh0 < 1e-8:
            break
    if False:
        pyplot.subplot(221)
        pyplot.plot(newVals)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(222)
        pyplot.plot(alphas)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(223)
        pyplot.plot(betas)
        pyplot.yscale('log')
        pyplot.xscale('log')
        pyplot.grid()
        pyplot.subplot(224)
        pyplot.plot(offsets())
        pyplot.grid()
        pyplot.show()
    print('Achieved {} in {} steps'.format(dnew/thresh0, i))

    offsetMap.clearmaps()
    offsetMap.binOffsets(offsets.offsets,
                         data.residual.wei,
                         offsets.offsetpixels,
                         data.pixels)
    offsetMap.average()
