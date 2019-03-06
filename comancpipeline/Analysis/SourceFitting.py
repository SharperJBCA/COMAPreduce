import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types
from scipy.optimize import fmin, leastsq
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust

from mpi4py import MPI 
comm = MPI.COMM_WORLD

import os

class JamesBeam:
    """
    Simple beam class that describes how the
    nominal beam solid angle changes with frequency
    based on James beam model.
    """
    def __init__(self):
        self.nu = np.array([26., 33., 40.])
        self.solidAngle = np.array([2.1842e-6, 1.6771e-6, 1.4828e-6])
        self.pmdl = interp1d(self.nu, self.solidAngle)

    def __call__(self, nu):
        return self.pmdl(nu)

beams = {'JamesBeam': JamesBeam}

def fisher(_x,_y,P):
    
    x, y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    x = x.flatten()
    y = y.flatten()

    A, sig, x0, y0, B = P
    r = (x - x0)**2 + (y - y0)**2

    f = np.exp(-0.5*r/sig**2)
    d0 = f
    d1 = r/sig**3 * f
    d2 = A * (x - x0)/sig**2 * f 
    d3 = A * (y - y0)/sig**2 * f
    d4 = np.ones(f.size)
    derivs = [d0, d1, d2,d3, d4]
    F = np.zeros((len(derivs), len(derivs)))
    for i in range(len(derivs)):
        for j in range(len(derivs)):
            F[i,j] = np.sum(derivs[i]*derivs[j])
    return F
class FitSource(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, filtertod=False):
        self.beamModel = beams[beammodel]()#getattr(__import__(__name__), beammodel)()
        self.x0 = x0
        self.y0 = y0
        self.lon = lon
        self.lat = lat 

        self.closeR = 12.5/60.
        self.filterel = True

        if isinstance(filtertod, str):
            self.filtertod = (filtertod.lower() == 'true')
        else:
            self.filtertod = filtertod

       
    def __str__(self):
        return 'Fitting source'

    def getHornRange(self,data):
        tod = data.getdset('spectrometer/tod')

        if (len(tod.shape) == 4) & (tod.shape[0] == 19):
            hornRange = range(0,tod.shape[0])
            mode = 1
        elif (len(tod.shape) < 4):
            hornRange = range(data.selectAxes['spectrometer/tod'][0], data.selectAxes['spectrometer/tod'][0]+1)
            mode = 2
        else:
            hornRange = range(data.lo['spectrometer/tod'], data.hi['spectrometer/tod'])
            mode = 3

        return hornRange, mode

    def run(self,data):
        self.fit(data)
        
    def initialPeak(self,tod, x, y):
        
        rms  = Filtering.calcRMS(tod)
        r = np.sqrt((x)**2 + (y)**2)
        close = (r < self.closeR)                      
        tod -= Filtering.estimateBackground(tod, rms, close)

        dx, dy = 1./60., 1./60.
        Dx, Dy = 1., 1.
        npix = int(Dx/dx)
        xpix, ypix = np.arange(npix+1), np.arange(npix+1)
        xpix = xpix*dx - Dx/2.
        ypix = ypix*dy - Dy/2.
        m = np.histogram2d(x, y, xpix, weights=tod)[0]/np.histogram2d(x, y, xpix)[0]
        m = median_filter(m, 3)
        try:
            xmax,ymax = np.unravel_index(np.nanargmax(m),m.shape)
        except ValueError:
            return None
        return xpix[xmax], ypix[ymax]

    def fit(self,data):
        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        nu  = data.getdset('spectrometer/frequency')
      
        rms  = Filtering.calcRMS(tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = tod.shape

        nParams = 5
        self.Pout = np.zeros((nHorns, nSBs, nChans, nParams))
        self.Perr = np.zeros((nHorns, nSBs, nChans, nParams))

        for i in range(nHorns):
            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

            good = (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)

            r = np.sqrt((x)**2 + (y)**2)

            if all(np.isnan(todAvg)):
                continue
            try:
                fitxy = self.initialPeak(todAvg[good], x, y)
            except IndexError:
                continue
            if isinstance(fitxy, type(None)):
                continue

            fitx0, fity0 = fitxy
            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    mask = np.isnan(todFit)
                    if all(mask):
                        continue
                    
                    if any(mask):
                        todFit[mask] = np.interp(np.flatnonzero(mask), 
                                                 np.flatnonzero(~mask), 
                                                 todFit[~mask])

                    close = (r < self.closeR)                      

                    if self.filtertod:
                        todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)
                    
                    P0 = [np.max(todFit) - np.median(todFit),
                          4./60./2.355,
                          fitx0,
                          fity0,
                          np.median(todFit)]

                    fout = leastsq(Fitting.ErrorLstSq, P0,
                                   Dfun = Fitting.DFuncLstSq,
                                   full_output=True, args=(Fitting.Gauss2d,
                                                           Fitting.Gauss2dLimits,
                                                           x, y, todFit, 0,0))


                    cov = fout[1]
                    if isinstance(cov, type(None)):
                        ferr = fout[0]*0.
                    else:
                        resid = np.std(todFit-Fitting.Gauss2d(fout[0], x, y,0,0))
                        cov *= resid**2
                        ferr = np.sqrt(np.diag(cov))

                    # print(fout[0],flush=True)
                    # print(ferr, flush=True)
                    # pyplot.plot(todFit, label='data')
                    # pyplot.plot(Fitting.Gauss2d(fout[0], x, y,0,0),'-k',linewidth=2,label='Best fit')
                    # pyplot.xlabel('Time')
                    # pyplot.ylabel('Antenna Temperature (K)')
                    # pyplot.title(comm.rank)
                    # pyplot.show()
                    # def bootstrap():
                    #     niter = 100
                    #     fouts = np.zeros((len(P0), niter))
                    #     todtmp = todFit[close]
                    #     xtmp = x[close]
                    #     ytmp = y[close]
                    #     select = np.random.uniform(low=0,high=todtmp.size,size=(niter,todtmp.size)).astype(int)
                    #     for iteration in range(niter):
                    #         fout = fmin(Fitting.ErrorFmin, P0, maxiter=100, maxfun=100,
                    #                     full_output=True, args=(Fitting.Gauss2d,
                    #                                             Fitting.Gauss2dLimits,
                    #                                             xtmp[select[iteration]], 
                    #                                             ytmp[select[iteration]], 
                    #                                             todtmp[select[iteration]], 0,0),
                    #             disp=False)

                    #         fouts[:,iteration] = fout[0]

                    #     return np.median(fouts,axis=1), robust.mad(fouts[0,:])

                    #fout, ferr = bootstrap()
                    #resid = todFit-Fitting.Gauss2d(fout, x, y,0,0)
                    #Cov   = fout[0][:,np.newaxis].dot(fout[0][np.newaxis,:]) / np.nanstd(resid)**2 
                    self.Perr[i,j,k,:] = ferr #np.sqrt(1./np.diag(Cov))
                    self.Pout[i,j,k,:] = fout[0]
                    #if comm.rank == 0:
                    #    for iteration in range(fout.size):
                    #        print(fout[iteration], ferr[iteration],flush=True)
                    #pyplot.plot(todFit-Fitting.Gauss2d(fout[0], x, y,0,0))
                    #pyplot.show()
                
        #nHorns, nSBs, nChans, nSamples = data.data['spectrometer/tod'].shape
        data.setextra('JupiterFits/Parameters', 
                      self.Pout,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_,
                       Types._OTHER_])
        data.setextra('JupiterFits/Uncertainties', 
                      self.Perr,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_,
                       Types._OTHER_])
        data.setextra('JupiterFits/frequency', 
                      nu,
                      [Types._SIDEBANDS_, 
                       Types._FREQUENCY_])

    def plot(self,data):

        if comm.rank > 0:
            return None

        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        nu  = data.getdset('spectrometer/frequency')
        rms  = Filtering.calcRMS(tod)

        nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)

            r = np.sqrt((x)**2 + (y)**2)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

            fitxy = self.initialPeak(todAvg[good], x, y)
            if isinstance(fitxy, type(None)):
                continue

            prefix = data.data.filename.split('/')[-1].split('.')[0]
            if not os.path.exists('Plotting/{}'.format(prefix)):
                os.makedirs('Plotting/{}'.format(prefix))
            
            fitx0, fity0 = fitxy
            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    close = (r < self.closeR)                      
                    if self.filtertod:
                        todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)

                    
                    pyplot.plot(todFit[close], label='data')
                    pyplot.plot(Fitting.Gauss2d(self.Pout[i,j,k,:], x[close], y[close],0,0),'-k',linewidth=2,label='Best fit')
                    pyplot.xlabel('Time')
                    pyplot.ylabel('Antenna Temperature (K)')
                    pyplot.savefig('Plotting/{}/JupiterFit{}{}{}.png'.format(prefix, i,j,k))
                    pyplot.clf()


class FitPlanet(FitSource):

    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, planet='jupiter', filtertod=False):
        super().__init__(beammodel, x0, y0, lon, lat, filtertod)
        self.planet = 'jupiter'

    def __str__(self):
        return 'Fitting {}'.format(self.planet)

    def getJupiter(self, data):
        mjd = data.getdset('spectrometer/MJD')
        self.x0, self.y0, self.dist = Coordinates.getPlanetPosition(self.planet, self.lon, self.lat, mjd)
        return self.x0, self.y0, self.dist

    def run(self, data):
        self.x0, self.y0, self.dist = self.getJupiter(data)
        self.fit(data)
    

class FitSourceApPhot(FitSource):

    def fit(self,data):
        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        nu  = data.getdset('spectrometer/frequency')

        maps = data.getextra('Mapping/SimpleMaps')
        self.wcs,_,_ = DefineWCS(naxis=[180,180], 
                                 cdelt=[1./60.,1./60.], 
                                 crval=[0,0],
                                 ctype=['RA---TAN','DEC--TAN'])

        
        rms  = Filtering.calcRMS(tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = tod.shape

        nParams = 1
        self.Pout = np.zeros((nHorns, nSBs, nChans, nParams))
        for i in range(nHorns):
            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)
            r = np.sqrt((x)**2 + (y)**2)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)
            fitx0, fity0 = self.initialPeak(todAvg[good], x, y)
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)

            pix = ang2pixWCS(self.wcs, x, y).astype('int')
            xr, yr = WCS.pix2ang1D(self.wcs,[180,180], pix)
            rpix = np.sqrt(xr**2 + yr**2)

            mask = np.where((pix != -1))[0]

            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    close = (r < self.closeR)                      

                    if self.filtertod:
                        todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)
                    
                    aperture = (rpix < 4./60.)
                    annulus  = (rpix > 5./60.) & (rpix < 6./60.)
                    
                    sl = maps[i,j,k,...]
                    apFlux = np.nansum(sl[aperture])
                    apN    = np.sum(aperture)
                    annuFlux = np.nanmean(sl[annulus])*apN
                    self.Pout[i,j,k,0] = apFlux - annuFlux
                                    
        print('writing extras')
        data.setextra('SourceFits/ApertureFlux', 
                      self.Pout,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_,
                       Types._OTHER_])

class FitPlanetApPhot(FitSourceApPhot):

    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, planet='jupiter', filtertod=False):
        super().__init__(beammodel, x0, y0, lon, lat, filtertod)
        self.planet = 'jupiter'

    def __str__(self):
        return 'Fitting {}'.format(self.planet)

    def getJupiter(self, data):
        mjd = data.getdset('spectrometer/MJD')
        self.x0, self.y0, self.dist = Coordinates.getPlanetPosition(self.planet, self.lon, self.lat, mjd)
        return self.x0, self.y0, self.dist

    def run(self, data):
        self.x0, self.y0, self.dist = self.getJupiter(data)
        self.fit(data)

