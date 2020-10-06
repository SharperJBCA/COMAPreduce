import numpy as np
from astropy import wcs
from matplotlib import pyplot
import h5py
from comancpieline.Tools import binFuncs
from scipy.interpolate import interp1d
import os

from matplotlib.patches import Ellipse

# Class for storing source locations
class Source:
    def __init__(self,x,y, hmsMode=True):
        """
        If hmsMode = True (default) then x and y are 3 element lists containing (HH,MM,SS.S) and (DD,MM,SS.S) respectively.
        else just pass the degrees of each coordinates
        """
        
        if hmsMode:
            self.x = self.HMS2Degree(x)
            self.y = self.DMS2Degree(y)
        else:
            self.x = x
            self.y = y

    def __call__(self):
        return self.x, self.y
    
    def DMS2Degree(self,d):
        """
        Convert DD:MM:SS.S format to degrees
        """
        return d[0] + d[1]/60. + d[2]/60.**2
    
    def HMS2Degree(self,d):
        return self.DMS2Degree(d)*15

sources = {'TauA':Source([5,34,31.94],[22,0,52.2]),
           'CasA':Source([23,23,24.0],[58,48,54.0]),
           'CygA':Source([19,59,28.36],[40,44,2.10])}


## -- Filters

class NormaliseFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        rms = np.nanstd(tod[1:tod.size//2*2:2] - tod[0:tod.size//2*2:2])
        tod = (tod - np.nanmedian(tod))/rms # normalise
        return tod

class AtmosphereFilter:
    def __init__(self,**kwargs):
        pass

    def __call__(self,DataClass, tod, **kwargs):
        feed = kwargs['FEED']
        el   = DataClass.el[feed,:]
        mask = DataClass.atmmask

        gd = (np.isnan(tod) == False) & (mask == 1)
        try:
            # Calculate slab
            A = 1./np.sin(el*np.pi/180.)
            # Build atmospheric model
            pmdl = np.poly1d(np.polyfit(A[gd],tod[gd],1))
            # Subtract atmospheric slab
            tod -= pmdl(A)

            # Bin by elevation, and remove with interpolation (redundant?) 
            binSize = 12./60.
            nbins = int((np.nanmax(el)-np.nanmin(el) )/binSize)
            elEdges= np.linspace(np.nanmin(el),np.nanmax(el),nbins+1)
            elMids = (elEdges[:-1] + elEdges[1:])/2.
            s = np.histogram(el[gd], elEdges, weights=tod[gd])[0]
            w = np.histogram(el[gd], elEdges)[0]
            pmdl = interp1d(elMids, s/w, bounds_error=False, fill_value=0)
            tod -= pmdl(el)
            tod[el < elMids[0]] -= s[0]/w[0]
            tod[el > elMids[-1]] -= s[-1]/w[-1]
        except TypeError:
            return tod 


        return tod
