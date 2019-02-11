import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

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

        self.fields = {'spectrometer/tod':True,
                       'spectrometer/MJD':False,
                       'spectrometer/pixel_pointing/pixel_ra':False,
                       'spectrometer/pixel_pointing/pixel_dec':False,
                       'spectrometer/pixel_pointing/pixel_el':False}

        
    def __str__(self):
        return 'Fitting source'

    def getHornRange(self,data):
        tod = data.dsets['spectrometer/tod']

        print(tod.shape)
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
        xmax,ymax = np.unravel_index(np.nanargmax(m),m.shape)
        return xpix[xmax], ypix[ymax]

    def fit(self,data):
        hornRange, mode = self.getHornRange(data)
        tod = data.dsets['spectrometer/tod']
        mjd = data.dsets['spectrometer/MJD']
        ra  = data.dsets['spectrometer/pixel_pointing/pixel_ra']
        dec = data.dsets['spectrometer/pixel_pointing/pixel_dec']
        el  = data.dsets['spectrometer/pixel_pointing/pixel_el']

        rms  = Filtering.calcRMS(tod)
        # loop over horns
        nHorns, nSBs, nChans, nSamples = tod.shape
        for i in hornRange:
            print(ra.shape, i)
            good = (np.isnan(ra[i,:]) == False) & (np.isnan(tod[i,0,0]) == False)
            pa = Coordinates.pa(ra[i,good], dec[i,good], mjd[good], self.lon, self.lat)
            x, y = Coordinates.Rotate(ra[i,good], dec[i,good], self.x0, self.y0, -pa)
            r = np.sqrt((x)**2 + (y)**2)

            todAvg = np.nanmean(np.nanmean(tod[i,...],axis=0),axis=0)
            fitx0, fity0 = self.initialPeak(todAvg[good], x, y)

            for j in range(nSBs):
                for k in range(nChans):
                    todFit = tod[i,j,k,good]
                    close = (r < self.closeR)                      

                    if self.filtertod:
                        todFit -= Filtering.estimateBackground(todFit, rms[i,j,k], close)
                    


                    P0 = [np.max(todFit) -np.median(todFit) ,
                          4./60./2.355,
                          fitx0,
                          fity0,
                          np.median(todFit)]
                                        
                    fout = fmin(Fitting.ErrorFmin, P0,full_output=True, args=(Fitting.Gauss2d,
                                                                                  Fitting.Gauss2dLimits,
                                                                                  x, y, todFit, 0,0), disp=False)

            

class FitPlanet(FitSource):

    def __init__(self, beammodel = 'JamesBeam', x0=0, y0=0, lon=-118.2941, lat=37.2314, planet='jupiter', filtertod=False):
        super().__init__(beammodel, x0, y0, lon, lat, filtertod)
        self.planet = 'jupiter'

    def __str__(self):
        return 'Fitting {}'.format(self.planet)

    def getJupiter(self, data):
        mjd = data.dsets['spectrometer/MJD']
        self.x0, self.y0, self.dist = Coordinates.getPlanetPosition(self.planet, self.lon, self.lat, mjd)
        return self.x0, self.y0, self.dist

    def run(self, data):
        self.x0, self.y0, self.dist = self.getJupiter(data)
        self.fit(data)
    
        
        
