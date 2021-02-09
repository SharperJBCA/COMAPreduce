import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools import  binFuncs, stats, Coordinates


class FlatMapType:

    def __init__(self,*args):
        self.setWCS(*args)

        self.sig = np.zeros(self.nypix*self.nxpix)
        self.wei = np.zeros(self.nypix*self.nxpix)
        self.hits= np.zeros(self.nypix*self.nxpix)

        self.sky_map = np.zeros(self.sig.size)
        self.cov_map = np.zeros(self.sig.size)
        self.hit_map = np.zeros(self.sig.size)

        self.npix = int(self.nypix*self.nxpix)

    def average(self):
        """
        Average the maps
        """
        gd = (self.wei != 0)
        self.sky_map[gd] = self.sig[gd]/self.wei[gd]
        self.cov_map[gd] = 1./self.wei[gd]
        self.hit_map[gd] = self.hits[gd]

    def get_map(self):
        """
        Return signal map weighted by covariance
        """
        
        self.average()
        
        return np.reshape(self.sky_map,(self.nypix,self.nxpix))

    def get_cov(self):
        """
        Return covariance map
        """
        self.average()

        return np.reshape(self.cov_map,(self.nypix,self.nxpix))

    def get_hits(self):
        """
        Return hit map distribution
        """
        self.average()
        return np.reshape(self.hit_map,(self.nypix,self.nxpix))

    def setWCS(self, *args):
        """
        Declare world coordinate system for plots
        """
        
        crval, cdelt, crpix, ctype,nxpix,nypix = args
        if isinstance(crval[0],str):
            crval[0] = Coordinates.sex2deg(crval[0],hours=True)
            crval[1] = Coordinates.sex2deg(crval[1],hours=False)

        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

        self.crval = self.wcs.wcs.crval
        self.cdelt = self.wcs.wcs.cdelt
        self.crpix = self.wcs.wcs.crpix
        self.ctype = self.wcs.wcs.ctype
        self.nxpix = nxpix
        self.nypix = nypix

    def getFlatPixels(self, x, y, return_xy=False):
        """
        Convert sky angles to pixel space
        """
        if isinstance(self.wcs, type(None)):
            raise TypeError( 'No WCS object declared')
            return
        else:
            pixels = self.wcs.wcs_world2pix(x+self.wcs.wcs.cdelt[0]/2.,
                                            y+self.wcs.wcs.cdelt[1]/2.,0)
            pflat = (pixels[0].astype(int) + self.nxpix*pixels[1].astype(int)).astype(int)
            

            # Catch any wrap around pixels
            pflat[(pixels[0] < 0) | (pixels[0] > self.nxpix)] = -1
            pflat[(pixels[1] < 0) | (pixels[1] > self.nypix)] = -1
        if return_xy:
            return pflat,pixels
        else:
            return pflat


    def sum_data(self,tod,pixels, weights=None,mask=None):
        """
        Sum data into map vectors
        """

        if isinstance(mask,type(None)):
            mask = np.ones(tod.size).astype(int)

        ones = np.ones(tod.size)
        if isinstance(weights,type(None)):
            binFuncs.binValues(self.sig  , pixels, weights=tod,mask=mask)
            binFuncs.binValues(self.wei  , pixels, weights=ones,mask=mask)
        else:
            binFuncs.binValues(self.sig  , pixels, weights=tod*weights,mask=mask)
            binFuncs.binValues(self.wei  , pixels, weights=weights,mask=mask)

        binFuncs.binValues(self.hits , pixels, weights=ones,mask=mask)


    def sum_offsets(self,offsets,weights,offsetpixels,pixels):
        """
        Add more data to the naive map
        """
        binFuncs.binValues2Map(self.sig, pixels, offsets*weights, offsetpixels)
        binFuncs.binValues2Map(self.wei, pixels, weights        , offsetpixels)

