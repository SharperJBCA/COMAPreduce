import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs
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

def makemap(d,x,y,ra0=0,dec0=0, cd=1./60., nxpix=600, nypix=400):

    xy = np.zeros((x.size,2))
    xy[:,0] = x.flatten()
    xy[:,1] = y.flatten()

    from astropy import wcs

    w = wcs.WCS(naxis=2)
    w.wcs.crval = [ra0, dec0]
    w.wcs.cdelt = [cd,cd]
    w.wcs.crpix = [nxpix/2., nypix/2.]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    pixels = w.wcs_world2pix(xy,0)
    ygrid, xgrid = np.meshgrid(np.arange(nypix),np.arange(nxpix))

    pixCens = w.wcs_pix2world(np.array([xgrid.flatten(), ygrid.flatten()]).T,0)
    pixCens[:,0] += 1./2.*cd
    pixCens[:,1] += 1./2.*cd
    pflat = (pixels[:,1].astype(int) + (nypix)*pixels[:,0].astype(int)).astype(int)


    pEdges = np.arange(nxpix*nypix+1)
    m = np.histogram(pflat,pEdges, weights=d)[0]
    h = np.histogram(pflat,pEdges)[0]
    m = m/h
    return m,w

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

    def getfwhm(self, nu):
        return np.sqrt(self.pmdl(nu)/1.13)*180./np.pi

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

        self.closeR = 20./60.
        self.vClose = 5./60.
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

        self.checkVane(data)
        self.fit(data)
        
    def checkVane(self,data):
        
        vane = data.getdset('hk/antenna0/vane/state')
        vutc = data.getdset('hk/antenna0/vane/utc')
        utc  = data.getdset('spectrometer/MJD')

        # interp vane data
        vaneFull = interp1d(vutc, vane, bounds_error=False, fill_value=0)(utc)
        data.dsets['spectrometer/vane']= vaneFull

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
            return 0,0
        return xpix[xmax], ypix[ymax]


    def fit(self,data):
        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        nu  = data.getdset('spectrometer/frequency')
        vane = data.getdset('spectrometer/vane')

        rms  = Filtering.calcRMS(tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = tod.shape

        nParams = 9
        self.Pout = np.zeros((nHorns, nSBs, nChans, nParams))
        self.Perr = np.zeros((nHorns, nSBs, nChans, nParams))
        peakEl = np.zeros((nHorns, nSBs, nChans))

        for i in range(nHorns):
            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

            good = (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            #x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)
            x, y = Coordinates.Rotate(ra[i,:], dec[i,:], self.x0, self.y0, 0)

            #print(select)
            #pyplot.plot(np.abs(dx))
            #pyplot.plot(select, np.abs(dx[select]),'-')
            #pyplot.show()


            if all(np.isnan(todAvg)):
                continue
            try:
                fitxy = self.initialPeak(todAvg[good], x[good], y[good])
            except IndexError:
                continue
            if isinstance(fitxy, type(None)):
                continue
            fitx0, fity0 = fitxy
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)

            good = (r < self.closeR) & (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, 0)

            close = (r < self.closeR)       
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
            vclose = (r < self.vClose)
            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    mask = np.isnan(todFit) | (vane[good] > 0)
                    if all(mask):
                        continue
                    
                    if any(mask):
                        todFit[mask] = np.interp(np.flatnonzero(mask), 
                                                 np.flatnonzero(~mask), 
                                                 todFit[~mask])

                    if self.filtertod:                      
                        pmdl = np.poly1d(np.polyfit(np.where(~vclose)[0], todFit[~vclose],5))
                        todFit -= pmdl(np.arange(todFit.size))

                    P0 = [np.max(todFit) - np.median(todFit),
                          fitx0 - 0/60.,
                          4./60./2.355,
                          fity0 - 1/60.,
                          9./60./2.355,
                          50*np.pi/180.,
                          np.median(todFit),
                          0.,
                          0.]

                    fout = leastsq(Fitting.ErrorLstSq, P0,
                                   Dfun = Fitting.DFuncGaussRotPlaneLstSq,
                                   full_output=True, 
                                   maxfev = 100,#*(len(P0) + 1),
                                   #ftol = 1e-15,
                                   args=(Fitting.Gauss2dRotPlane,
                                         Fitting.Gauss2dRotPlaneLimits,
                                         x, y, todFit, 0,0))
                    #print(fout)
                    fout[0][5] = np.mod(fout[0][5], 2*np.pi)
                    cov = fout[1]
                    if isinstance(cov, type(None)):
                        ferr = fout[0]*0.
                    else:
                        resid = np.std(todFit-Fitting.Gauss2d(fout[0], x, y,0,0))
                        cov *= resid**2
                        ferr = np.sqrt(np.diag(cov))


                    # Format values:
                    #pyplot.plot(todFit, label='data',zorder=0)
                    #pyplot.plot(Fitting.Gauss2dRotPlane(fout[0], x, y,0,0),'--r',linewidth=2,label='Best fit',zorder=2)
                    if fout[0][2] > fout[0][4]: # want x to be smaller than y
                        _temp = fout[0][2]*1.
                        fout[0][2] = fout[0][4]*1.
                        fout[0][4] = _temp
                        fout[0][5] = np.mod(fout[0][5] - np.pi/2., np.pi)
                    else:
                        fout[0][5] = np.mod(fout[0][5], np.pi)
                    #print(fout[0][5]*180./np.pi)
                    #pyplot.plot(Fitting.Gauss2dRotPlane(P0, x, y,0,0),'-k',linewidth=2,label='Best fit',zorder=1)
                    #pyplot.show()

                    self.Perr[i,j,k,:] = ferr #np.sqrt(1./np.diag(Cov))
                    self.Pout[i,j,k,:] = fout[0]

                    bestfit = Fitting.Gauss2dRotPlane(self.Pout[i,j,k,:],
                                                      x, 
                                                      y,0,0)
                    peakEl[i,j,k]=el[i,np.argmax(bestfit)]

        #nHorns, nSBs, nChans, nSamples = data.data['spectrometer/tod'].shape
        data.setextra('FitSource/Parameters', 
                      self.Pout,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_,
                       Types._OTHER_])
        data.setextra('FitSource/Uncertainties', 
                      self.Perr,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_,
                       Types._OTHER_])
        data.setextra('FitSource/frequency', 
                      nu,
                      [Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('JupiterFits/peakel', 
                      peakEl,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
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
        vane = data.getdset('spectrometer/vane')

        nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, 0)

            r = np.sqrt((x)**2 + (y)**2)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

            try:
                fitxy = self.initialPeak(todAvg[good], x, y)
            except IndexError:
                continue
            if isinstance(fitxy, type(None)):
                continue

            prefix = data.data.filename.split('/')[-1].split('.')[0]
            if not os.path.exists('Plotting/{}'.format(prefix)):
                os.makedirs('Plotting/{}'.format(prefix))
            
            fitx0, fity0 = fitxy
            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    mask = np.isnan(todFit) | (vane[good] > 0)
                    if all(mask):
                        continue
                    
                    if any(mask):
                        todFit[mask] = np.interp(np.flatnonzero(mask), 
                                                 np.flatnonzero(~mask), 
                                                 todFit[~mask])

                    close = (r < self.closeR)  
                    vclose = (r < self.vClose)

                    if self.filtertod:
                        #todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)
                        pmdl = np.poly1d(np.polyfit(np.where(~vclose)[0], todFit[~vclose],5))
                        todFit -= pmdl(np.arange(todFit.size))

                    
                    pyplot.plot(todFit[close], label='data')
                    pyplot.plot(Fitting.Gauss2dRotPlane(self.Pout[i,j,k,:], x[close], y[close],0,0),'-k',linewidth=2,label='Best fit')
                    pyplot.xlabel('Time')
                    pyplot.ylabel('Antenna Temperature (K)')
                    pyplot.savefig('Plotting/{}/SourceFit{}{}{}.png'.format(prefix, i,j,k))
                    #pyplot.show()
                    pyplot.clf()


class FitSourceAlternateScans(FitSource):
    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, filtertod=False):
        super().__init__(beammodel, x0, y0, lon, lat, filtertod)

    def scanEdges(self,x, idir=0):
        """
        Calculate edges of raster scans.
        """
        assert idir < 2

        #dx = np.gradient(np.gradient(x))
        rms = np.std(x[1::2] - x[:-1:2]) /np.sqrt(2)
        cutdx = rms*3

        edges = ffuncs.scanedges(x, rms)
        edges = edges[edges != 0]
        diff = (edges[1:] - edges[:-1])
        bad = np.where(diff < 5)[0]
        edges = np.delete(edges, bad)

        #peaks = np.concatenate((np.where((np.abs(dx) > cutdx))[0], np.array([dx.size])))

        #sepPeaks = peaks[1:] - peaks[:-1]
        #whePeaks = np.where(sepPeaks > 300)[0]
        scanEdges = edges.astype(int) #peaks[whePeaks]
        # d1 scans:
        select = np.zeros(x.size).astype(bool)
        #pyplot.plot(x)

        time = np.arange(x.size)
        for pid in range(scanEdges.size-1):

            try:
                dx = np.gradient(x[scanEdges[pid]:scanEdges[pid+1]])
            except ValueError:
                print(scanEdges[pid], scanEdges[pid+1])
                pyplot.plot(x)
                pyplot.show()

            if (idir ==0) & (x[scanEdges[pid]] < x[scanEdges[pid+1]-1]): #CW
                select[scanEdges[pid]:scanEdges[pid+1]] = True
            elif (idir == 1) & (x[scanEdges[pid]] > x[scanEdges[pid+1]-1]): #CCW
                select[scanEdges[pid]:scanEdges[pid+1]] = True

        return select

    def fit(self,data):
        self.lon = data.getdset('hk/antenna0/tracker/siteActual')[0,0]/(60.*60.*1000.)
        self.lat = data.getdset('hk/antenna0/tracker/siteActual')[0,1]/(60.*60.*1000.)

        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        nu  = data.getdset('spectrometer/frequency')
        az  = data.getdset('spectrometer/pixel_pointing/pixel_az')

        rms  = Filtering.calcRMS(tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = tod.shape

        nParams = 9
        self.Pout = np.zeros((nHorns, nSBs, nChans, 2, nParams))
        self.Perr = np.zeros((nHorns, nSBs, nChans, 2, nParams))
        peakEl = np.zeros((nHorns, nSBs, nChans, 2))
        peakAz = np.zeros((nHorns, nSBs, nChans, 2))
        peakMJD = np.zeros((nHorns, nSBs, nChans, 2))

        for i in range(nHorns):
            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

            good = (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
            pa = Coordinates.pa(ra[i,:], dec[i,:], mjd[:], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,:], dec[i,:], self.x0, self.y0, -pa)


            if all(np.isnan(todAvg)):
                print('TOD ALL NAN')
                continue
            try:
                fitxy = self.initialPeak(todAvg[good], x[good], y[good])
            except IndexError:
                fitxy = None

            if isinstance(fitxy, type(None)):
                print('FITXY IS NONE')
                continue

            fitx0, fity0 = fitxy
            r = np.sqrt((x)**2 + (y)**2)

            good = (r < self.closeR) & (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa[good])
            selects = [self.scanEdges(x, idir) for idir in range(2)]

            close = (r < self.closeR)       
            r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
            vclose = (r < self.vClose)

            close = (r < self.closeR)  
            for j in range(nSBs):

                for k in range(nChans):
                    todFit = tod[i,j,k,good]

                    time = np.arange(x.size)
                    colors = ['r','g']
                    titles = ['Forward', 'Backward']
                    for idir in range(2): # fit each direction in az seperately
                        select = selects[idir]
                        
                        mask = np.isnan(todFit)
                        if all(mask):
                            print('ALL NaN Values in todFit')
                            continue
                    
                        if any(mask):
                            todFit[mask] = np.interp(np.flatnonzero(mask), 
                                                     np.flatnonzero(~mask), 
                                                     todFit[~mask])


                        if self.filtertod:                        
                            pmdl = np.poly1d(np.polyfit(np.where(~vclose)[0], todFit[~vclose],1))
                            todFit -= pmdl(np.arange(todFit.size))



                        P0 = [np.max(todFit) - np.median(todFit),
                              fitx0,
                              4./60./2.355,
                              fity0,
                              4./60./2.355,
                              np.pi/2.,
                              np.median(todFit),
                              0.,
                              0.]

                        #pyplot.plot(x[select],  todFit[select])
                        #pyplot.show()
                        fout = leastsq(Fitting.ErrorLstSq, P0,
                                       Dfun = Fitting.DFuncGaussRotPlaneLstSq,
                                       full_output=True, 
                                       maxfev = 100,
                                       args=(Fitting.Gauss2dRotPlane,
                                             Fitting.Gauss2dRotPlaneLimits,
                                             x[select], y[select], todFit[select], 0,0))

                        fout[0][5] = np.mod(fout[0][5], 2*np.pi)
                        cov = fout[1]
                        if isinstance(cov, type(None)):
                            ferr = fout[0]*0.
                        else:
                            resid = np.std(todFit[select]-Fitting.Gauss2d(fout[0], x, y,0,0)[select])
                            cov *= resid**2
                            ferr = np.sqrt(np.diag(cov))

                        nx, ny = 50, 50

                        m, w = makemap(todFit[select],x[select],y[select], nxpix=nx,nypix=ny, cd=1.5/60.)

                        # Format values:
                        # pyplot.plot(todFit, label='data',zorder=0)
                        # pyplot.plot(Fitting.Gauss2dRotPlane(fout[0], x, y,0,0),'--r',linewidth=2,label='Best fit',zorder=2)
                        if fout[0][2] > fout[0][4]: # want x to be smaller than y
                            _temp = fout[0][2]*1.
                            fout[0][2] = fout[0][4]*1.
                            fout[0][4] = _temp
                            fout[0][5] = np.mod(fout[0][5] - np.pi/2., np.pi)
                        else:
                            fout[0][5] = np.mod(fout[0][5], np.pi)
                        #p#rint(fout[0][5]*180./np.pi)
                        self.Perr[i,j,k,idir,:] = ferr #np.sqrt(1./np.diag(Cov))
                        self.Pout[i,j,k,idir,:] = fout[0]

                        # BEST FIT MODEL
                        bestfit = Fitting.Gauss2dRotPlane(self.Pout[i,j,k,idir,:],
                                                          x, 
                                                          y,0,0)
                        peakEl[i,j,k,idir]=el[i,np.argmax(bestfit)]
                        peakAz[i,j,k,idir]=az[i,np.argmax(bestfit)]
                        peakMJD[i,j,k,idir]=mjd[np.argmax(bestfit)]

                        # pyplot.plot(Fitting.Gauss2dRotPlane(P0, x, y,0,0),'-k',linewidth=2,label='Best fit',zorder=1)
                        # pyplot.show()

                        if (j ==0) & (k==0):
                            pyplot.subplot(2,2,1+idir,projection=w)
                            pyplot.imshow(m.reshape(nx,ny).T, origin='lower', vmin=0)
                            pyplot.colorbar()
                            pyplot.xlabel(r'$\Delta \alpha$')
                            pyplot.ylabel(r'$\Delta \delta$')
                            pyplot.title(titles[idir])
                            pyplot.grid()
                    if (j ==0) & (k==0):
                        nx, ny = 50, 50

                        m, w = makemap(todFit,x,y, nxpix=nx,nypix=ny, cd=1.5/60.)

                        pyplot.subplot(2,2,1+2,projection=w)
                        pyplot.imshow(m.reshape(nx,ny).T, origin='lower', vmin=0)
                        pyplot.colorbar()
                        pyplot.xlabel(r'$\Delta \alpha$')
                        pyplot.ylabel(r'$\Delta \delta$')
                        pyplot.title('Combined')
                        pyplot.grid()
                        m, w = makemap(Fitting.Gauss2dRotPlane(fout[0], x, y,0,0),x,y, nxpix=nx,nypix=ny, cd=1.5/60.)

                        pyplot.subplot(2,2,1+3,projection=w)
                        pyplot.imshow(m.reshape(nx,ny).T, origin='lower', vmin=0)
                        pyplot.colorbar()
                        pyplot.xlabel(r'$\Delta \alpha$')
                        pyplot.ylabel(r'$\Delta \delta$')
                        pyplot.title('Fitted (backward)')
                        pyplot.grid()
                        prefix = data.data.filename.split('/')[-1].split('.')[0]
                        pyplot.tight_layout()
                        pyplot.savefig('TauA_{}_Horn0.png'.format(prefix),bbox_inches='tight')
                        pyplot.clf()

            data.setextra('JupiterFits/Parameters', 
                          self.Pout,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_,
                           Types._OTHER_])
            data.setextra('JupiterFits/Uncertainties', 
                          self.Perr,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_,
                           Types._OTHER_])
            data.setextra('JupiterFits/frequency', 
                          nu,
                          [Types._SIDEBANDS_, 
                           Types._FREQUENCY_])
            data.setextra('JupiterFits/peakel', 
                          peakEl,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/peakaz', 
                          peakAz,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/peakmjd', 
                          peakMJD,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/dT', 
                          (peakMJD-mjd[0])*86400.,
                          [Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])



    def plot(self,data):

        if comm.rank > 0:
            return None

        if True:
            return

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
                    colors = [(0.8,0.3,0.2),(0.2,0.8,0.3)]
                    colordata = [(0.9,0.5,0), (0,0.9,0.5)]
                    r = np.sqrt((x-self.Pout[i,j,k,0,1])**2 + (y-self.Pout[i,j,k,0,3])**2)

                    for idir in range(2): # fit each direction in az seperately

                        select = self.scanEdges(x, idir)

                        todFit = tod[i,j,k,good]*1.
                        close = (r[select] < self.closeR)    
                        
                        if self.filtertod:
                            #todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)
                            pmdl = np.poly1d(np.polyfit(np.where(~close)[0], (todFit[select])[~close],1))
                            #print(select, np.arange(select.size))
                            todFit[select] -= pmdl(np.where(select)[0])

                    
                        pyplot.plot(x[select[close]]*60, todFit[select[close]], label='data {}'.format(idir), color=colordata[idir])
                        pyplot.plot(x[select[close]]*60, Fitting.Gauss2dRotPlane(self.Pout[i,j,k,idir,:],
                                                            x[select[close]], 
                                                            y[select[close]],0,0),'--',
                                    linewidth=2,label='Best fit {}'.format(idir), color=colors[idir],zorder=10)
                    pyplot.xlabel('Azimuth Offset (arcmin)')
                    pyplot.ylabel('Antenna Temperature (K)')
                    pyplot.legend(loc='best', prop={'size':10})
                    pyplot.savefig('Plotting/{}/JupiterFit-SepScans_{:02d}-{:02d}-{:02d}.png'.format(prefix, i,j,k))
                    pyplot.clf()


class FitMultiSourceAlternateScans(FitSourceAlternateScans):
    """
    If there are multiple passes of a point source seperated by calibration loads then this
    method will first split each scan up, and fit for each pass over the source.

    Example observation ids: 6156, 6157, 6172, 6190, 6191 - Tau A raster observations to assess timing issue
    
    """
    
    def vaneEdges(self, x, mindist = 5000):
        """
        Find samples where the vane is in
        """
        
        vaneIn = np.where(x > 0)[0]
        dVaneIn = vaneIn[1:] - vaneIn[:-1]
        # anywhere dVaneIn is > 0 is a transition point
        #remove points where the separation is less the mindist
        outVanePos = []
        #if x[0] == 0:
        #    outVanePos += [0]

        for i in range(dVaneIn.size):
            if dVaneIn[i] > mindist:
                outVanePos += [[vaneIn[i],vaneIn[i+1]]]

        #if x[-1] == 0:
        #    outVanePos += [x.size]

        return outVanePos


    def fit(self,data):        

        self.lon = data.getdset('hk/antenna0/tracker/siteActual')[0,0]/(60.*60.*1000.)
        self.lat = data.getdset('hk/antenna0/tracker/siteActual')[0,1]/(60.*60.*1000.)

        _tod = data.getdset('spectrometer/tod')
        _mjd = data.getdset('spectrometer/MJD')
        _ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        _dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')
        _el  = data.getdset('spectrometer/pixel_pointing/pixel_el')
        _nu  = data.getdset('spectrometer/frequency')
        _az  = data.getdset('spectrometer/pixel_pointing/pixel_az')

        vaneUTC   = data.getdset('hk/antenna0/vane/utc')
        vaneState = data.getdset('hk/antenna0/vane/state')
        vaneState = interp1d(vaneUTC, vaneState, fill_value=0, bounds_error=False)(_mjd)
        scanRanges = self.vaneEdges(vaneState)
        nPasses = len(scanRanges)
        rms  = Filtering.calcRMS(_tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = _tod.shape

        nParams = 9
        self.Pout = np.zeros((nPasses, nHorns, nSBs, nChans, 2, nParams))
        self.Perr = np.zeros((nPasses, nHorns, nSBs, nChans, 2, nParams))
        peakEl    = np.zeros((nPasses, nHorns, nSBs, nChans, 2))
        peakAz    = np.zeros((nPasses, nHorns, nSBs, nChans, 2))
        peakMJD   = np.zeros((nPasses, nHorns, nSBs, nChans, 2))


        prefix = data.data.filename.split('/')[-1].split('.')[0]
        if not os.path.exists('Plotting/{}'.format(prefix)):
            os.makedirs('Plotting/{}'.format(prefix))
            

        for ipass, (t0, t1) in enumerate(scanRanges):
            tod = _tod[...,t0:t1]
            mjd = _mjd[...,t0:t1]
            ra  = _ra[...,t0:t1]
            dec =_dec[...,t0:t1]
            el  = _el[...,t0:t1]
            nu  = _nu[...,t0:t1]
            az  = _az[...,t0:t1]

            for i in range(nHorns):
                todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)

                good = (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
                pa = Coordinates.pa(ra[i,:], dec[i,:], mjd[:], self.lon, self.lat)
                x, y = Coordinates.Rotate(ra[i,:], dec[i,:], self.x0, self.y0, -pa)

                if all(np.isnan(todAvg)):
                    print('TOD ALL NAN')
                    continue
                try:
                    fitxy = self.initialPeak(todAvg[good], x[good], y[good])
                except IndexError:
                    fitxy = None

                if isinstance(fitxy, type(None)):
                    print('FITXY IS NONE')
                    continue

                fitx0, fity0 = fitxy
                r = np.sqrt((x)**2 + (y)**2)

                good = (r < self.closeR) & (np.isnan(ra[i,:]) == False) & (np.isnan(todAvg) == False)
                x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa[good])
                selects = [self.scanEdges(x, idir) for idir in range(2)]
                
                close = (r < self.closeR)       
                r = np.sqrt((x-fitx0)**2 + (y-fity0)**2)
                vclose = (r < self.vClose)

                close = (r < self.closeR)  
                for j in range(nSBs):

                    for k in range(nChans):
                        todFit = tod[i,j,k,good]
                        
                        time = np.arange(x.size)
                        colors = ['r','g']
                        titles = ['Forward', 'Backward']
                        for idir in range(2): # fit each direction in az seperately
                            select = selects[idir]
                        
                            mask = np.isnan(todFit)
                            if all(mask):
                                print('ALL NaN Values in todFit')
                                continue
                    
                            if any(mask):
                                todFit[mask] = np.interp(np.flatnonzero(mask), 
                                                         np.flatnonzero(~mask), 
                                                         todFit[~mask])


                            if self.filtertod:   
                                from scipy.signal import medfilt
                                stepSize = int(50*10)
                                nSteps = int(todFit.size//stepSize)
                                indices = np.arange(todFit.size).astype(int)
                                for istep in range(nSteps):
                                    if istep < nSteps-1:
                                        hi = (istep+1)*stepSize
                                    else:
                                        hi = todFit.size
                                        
                                    notClose = (indices[istep*stepSize:hi])[~vclose[istep*stepSize:hi]]

                                    pmdl = np.poly1d(np.polyfit(indices[notClose], todFit[notClose],1))
                                    todFit[istep*stepSize:hi] -= pmdl(indices[istep*stepSize:hi]) #np.median(todFit[notClose])
                            
                                #bkgd = todFit[~vclose]
                                #pmdl = np.poly1d(np.polyfit(np.where(~vclose)[0], todFit[~vclose],1))
                                #todFit -= pmdl(np.arange(todFit.size))
                            #pyplot.plot(todFit)
                            #pyplot.show()


                            P0 = [np.max(todFit) - np.median(todFit),
                                  fitx0,
                                  4./60./2.355,
                                  fity0,
                                  4./60./2.355,
                                  np.pi/2.,
                                  np.median(todFit),
                                  0.,
                                  0.]

                            #pyplot.plot(x[select],  todFit[select])
                            #pyplot.show()
                            fout = leastsq(Fitting.ErrorLstSq, P0,
                                           Dfun = Fitting.DFuncGaussRotPlaneLstSq,
                                           full_output=True, 
                                           maxfev = 100,
                                           args=(Fitting.Gauss2dRotPlane,
                                                 Fitting.Gauss2dRotPlaneLimits,
                                                 x[select], y[select], todFit[select], 0,0))

                            fout[0][5] = np.mod(fout[0][5], 2*np.pi)
                            cov = fout[1]
                            if isinstance(cov, type(None)):
                                ferr = fout[0]*0.
                            else:
                                resid = np.std(todFit[select]-Fitting.Gauss2d(fout[0], x, y,0,0)[select])
                                cov *= resid**2
                                ferr = np.sqrt(np.diag(cov))


                            # Format values:
                            if fout[0][2] > fout[0][4]: # want x to be smaller than y
                                _temp = fout[0][2]*1.
                                fout[0][2] = fout[0][4]*1.
                                fout[0][4] = _temp
                                fout[0][5] = np.mod(fout[0][5] - np.pi/2., np.pi)
                            else:
                                fout[0][5] = np.mod(fout[0][5], np.pi)

                            self.Perr[ipass,i,j,k,idir,:] = ferr #np.sqrt(1./np.diag(Cov))
                            self.Pout[ipass,i,j,k,idir,:] = fout[0]
                            
                            # BEST FIT MODEL
                            bestfit = Fitting.Gauss2dRotPlane(self.Pout[ipass,i,j,k,idir,:],
                                                              x, 
                                                              y,0,0)
                            peakEl[ipass,i,j,k,idir]=el[i,np.argmax(bestfit)]
                            peakAz[ipass,i,j,k,idir]=az[i,np.argmax(bestfit)]
                            peakMJD[ipass,i,j,k,idir]=mjd[np.argmax(bestfit)]

                            # pyplot.plot(todFit[select])
                            # pyplot.plot(Fitting.Gauss2dRotPlane(self.Pout[ipass,i,j,k,idir,:],x[select],y[select],0,0))
                            # if idir == 0:
                            #     pyplot.savefig('Plotting/{}/JupiterFit_Pass-{}_SepScans_{:02d}-{:02d}-{:02d}-CW.png'.format(prefix,ipass, i,j,k))
                            # else:
                            #     pyplot.savefig('Plotting/{}/JupiterFit_Pass-{}_SepScans_{:02d}-{:02d}-{:02d}-CCW.png'.format(prefix,ipass, i,j,k))

                            # pyplot.clf()


            data.setextra('JupiterFits/Parameters', 
                          self.Pout,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_,
                           Types._OTHER_])
            data.setextra('JupiterFits/Uncertainties', 
                          self.Perr,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_,
                           Types._OTHER_])
            data.setextra('JupiterFits/frequency', 
                          nu,
                          [Types._SIDEBANDS_, 
                           Types._FREQUENCY_])
            data.setextra('JupiterFits/peakel', 
                          peakEl,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/peakaz', 
                          peakAz,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/peakmjd', 
                          peakMJD,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])
            data.setextra('JupiterFits/dT', 
                          (peakMJD-mjd[0])*86400.,
                          [Types._OTHER_,
                           Types._HORNS_, 
                           Types._SIDEBANDS_, 
                           Types._FREQUENCY_,
                           Types._OTHER_])


class FitPlanet(FitSource):

    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, planet='jupiter', filtertod=False):
        super().__init__(beammodel, x0, y0, lon, lat, filtertod)
        self.planet = 'jupiter'

    def __str__(self):
        return 'Fitting {}'.format(self.planet)

    def getJupiter(self, data):
        self.lon = data.getdset('hk/antenna0/tracker/siteActual')[0,0]/(60.*60.*1000.)
        self.lat = data.getdset('hk/antenna0/tracker/siteActual')[0,1]/(60.*60.*1000.)

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

