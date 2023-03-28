#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:50:48 2023

@author: sharper
"""

import numpy as np
from tqdm import tqdm

from dataclasses import dataclass, field 
from .Running import PipelineFunction
from .DataHandling import HDF5Data , COMAPLevel2

from scipy.optimize import minimize
import logging 

from comancpipeline.Tools.median_filter import medfilt
from comancpipeline.Tools import Coordinates, binFuncs

from astropy.wcs import WCS
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass 
class SkyMap:

    _sky_sum : np.ndarray[float] = field(default_factory=lambda: None) # sky map summed data
    _wei_sum : np.ndarray[float] = field(default_factory=lambda: None) # sum weights map 
    _hit_sum : np.ndarray[float] = field(default_factory=lambda: None) # sum hits map 
        
    wcs : WCS = field(default_factory=lambda: None)
    x_npix : int = None
    y_npix : int = None

    def __post_init__(self):
        if (self.x_npix != None) & (self.y_npix != None):
            self.npix = self.x_npix * self.y_npix
            self._sky_sum = np.zeros(self.npix)
            self._wei_sum = np.zeros(self.npix)
            
    def set_map_info(self, x_npix, y_npix,
                     cdelt=None, 
                     crval=None, 
                     ctype=None,
                     crpix=None): 
        self.x_npix = x_npix 
        self.y_npix = y_npix 
        self.__post_init__() 
        
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crval = crval
        self.wcs.wcs.ctype = ctype 
        
        header = self.wcs.to_header()
        header['NAXIS1'] = self.x_npix 
        header['NAXIS2'] = self.y_npix 
        self.wcs = WCS(header) 
        
    def world_to_pixels(self, x : np.ndarray[float], y: np.ndarray[float]):
        """Convert coordinates to pixel positions""" 
        rows, columns = self.wcs.world_to_array_index_values(x,y)
        ipix = columns * self.y_npix + rows 
        
        bad = (ipix < 0) | (ipix >= self.npix) 
        return ipix.astype(int), bad

    def bin_data(self, 
                 tod : np.ndarray[float],
                 x : np.ndarray[float],
                 y : np.ndarray[float],
                 weights : np.ndarray[float] = None,
                 mask : np.ndarray[bool] = None): 
        """Bin data into the sky map arrays"""
        
        if isinstance(weights,type(None)): weights = np.ones(tod.size) 
        if isinstance(mask,type(None)): mask = np.ones(tod.size, dtype=bool) 

        pixels, pixel_mask = self.world_to_pixels(x,y)
        
        mask[pixel_mask] = False
        mask[(tod == 0) | ~np.isfinite(tod)] = False
        
        binFuncs.binValues(self._sky_sum,
                           pixels,
                           weights=tod*weights,
                           mask=mask.astype(int))
        binFuncs.binValues(self._wei_sum,
                           pixels,
                           weights=weights,
                           mask=mask.astype(int))
        
        i,j = self.wcs.world_to_array_index_values(0,0)
        rows, columns = self.wcs.world_to_array_index_values(x,y)

    @property 
    def world_flat(self):
        ypix,xpix = np.meshgrid(np.arange(self.y_npix),
                                np.arange(self.x_npix))
    
        x, y = self.wcs.array_index_to_world_values(ypix.flatten(),
                                                    xpix.flatten())
        
        return x, y

    @property 
    def m_flat(self):
        m = np.zeros(self.npix)
        good = (self._wei_sum != 0)
        
        m[good] = self._sky_sum[good]/self._wei_sum[good]
        
        return m

    @property 
    def m(self):
        
        return self.m_flat.reshape((self.x_npix,self.y_npix)).T
    
    @property 
    def variance_flat(self):
        m = np.zeros(self.npix)
        good = (self._wei_sum != 0)
        
        m[good] = 1./self._wei_sum[good]
        
        return m

    @property 
    def variance(self):
        
        return self.variance_flat.reshape((self.y_npix,self.x_npix)).T

    @property 
    def weights_flat(self):
        m = np.zeros(self.npix)
        good = (self._wei_sum != 0)
        
        m[good] = self._wei_sum[good]
        
        return m

    @property 
    def weights(self):
        
        return self.weights_flat.reshape((self.y_npix,self.x_npix)).T

@dataclass 
class SourcePosition:
    
    source : str = ''
    _mjd :np.ndarray = field(default_factory=lambda : np.zeros(1))

    _ra :np.ndarray= field(default_factory=lambda : np.zeros(1))
    _dec :np.ndarray= field(default_factory=lambda : np.zeros(1))
    _az :np.ndarray= field(default_factory=lambda : np.zeros(1))
    _el :np.ndarray= field(default_factory=lambda : np.zeros(1))
    mask :np.ndarray= field(default_factory=lambda : None)
    
    def __post_init__(self):
        self.az, self.el, self.ra, self.dec = Coordinates.sourcePosition(self.source, 
                                                                         self.mjd, 
                                                                         Coordinates.comap_longitude, 
                                                                         Coordinates.comap_latitude)

    
    @property
    def ra(self):
        if not isinstance(self.mask,type(None)):
            return self._ra[self.mask] 
        else:
            return self._ra 
    @ra.setter 
    def ra(self, v):
        self._ra = v 
        
    @property
    def dec(self):
        if not isinstance(self.mask,type(None)):
            return self._dec[self.mask] 
        else:
            return self._dec
    @dec.setter 
    def dec(self, v):
        self._dec = v 

    @property
    def az(self):
        if not isinstance(self.mask,type(None)):
            return self._az[self.mask] 
        else:
            return self._az
    @az.setter 
    def az(self, v):
        self._az = v 
        
    @property
    def el(self):
        if not isinstance(self.mask,type(None)):
            return self._el[self.mask] 
        else:
            return self._el
    @el.setter 
    def el(self, v):
        self._el = v 
        
    @property
    def mjd(self):
        if not isinstance(self.mask,type(None)):
            return self._mjd[self.mask] 
        else:
            return self._mjd
    @mjd.setter 
    def mjd(self, v):
        self._mjd = v 

        
    def rotate(self, x,y, theta):
        """Rotate angles by theta"""
        _x = x*np.cos(theta) + y*np.sin(theta)
        _y =-x*np.sin(theta) + y*np.cos(theta)
        
        return _x, _y 
    
    def rotate_ra(self, ra_telescope, dec_telescope): 
        pa = Coordinates.pa(self.ra, self.dec, self.mjd,
                            Coordinates.comap_longitude, 
                            Coordinates.comap_latitude)
        
        
        ra_r, dec_r = Coordinates.Rotate(ra_telescope,
                                         dec_telescope,
                                         self.ra, self.dec, -pa)
        
        s = SourcePosition(source=self.source, _mjd=self._mjd)
        s.ra = np.zeros(self._ra.size)
        s.ra[self.mask] = ra_r 
        s.dec = np.zeros(self._dec.size)
        s.dec[self.mask] = dec_r 
        s.set_mask(self.mask)
        return s 
        
    
    def set_mask(self, mask : np.ndarray[bool]):
        self.mask = mask 
        
@dataclass 
class FitSource(PipelineFunction):
    name : str = 'FitSource'
    
    overwrite : bool = False
    calibration : str = 'none' 
    STATE : bool = True 
    MEDIAN_FILTER_STEP : int = 1000
    NPARAMS : int = 7
    NXPIX : int = 100
    NYPIX : int = 100 
    
    source_position : SourcePosition = field(default_factory=lambda : None )

    def __post_init__(self):
        
        self.data = {}

        self.groups = [f'{self.calibration}_source_fit/fits',
                       f'{self.calibration}_source_fit/errors']

    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""


        return self.data, {}
    
    def __call__(self, data : HDF5Data, level2_data : COMAPLevel2): 
                
        self.fit_source(data, level2_data)
        
        return self.STATE 

    @staticmethod 
    def median_filter(tod,medfilt_stepsize):
        """
        Calculate this AFTER removing the atmosphere.
        """
        if any(~np.isfinite(tod)):
            return np.zeros(tod.size)
        if tod.size < medfilt_stepsize:
            return np.zeros(tod.size) + np.nanmedian(tod)
        filter_tod = np.array(medfilt.medfilt(tod.astype(np.float64),np.int32(medfilt_stepsize)))
        
        return filter_tod[:tod.size]
    
    
    def source_good(self):
        """Check if this is a valid source to fit"""
        if self.source in Coordinates.CalibratorList:
            logging.info(f'{self.source} found in Coordinates.CalibratorList')
            return True 
        else:
            logging.info(f'{self.source} NOT found in Coordinates.CalibratorList')
            return False

    def fit(self, 
            source_map : SkyMap):
        
        def dA(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            
            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  
                 
            mdl = np.exp(-X**2 - Y**2)
            return mdl 
        def dB(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat                 
            mdl = np.ones(x.size)
            return mdl 
        def dx0(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  

            part1 = 2*(sigma_x - sigma_y)*(sigma_x + sigma_y) * np.sin(theta)*np.cos(theta)*(y-y0)/(sigma_x*sigma_y)**2 
            part2 = 2 * np.cos(theta)**2*(x - x0)/sigma_x**2 
            part3 = 2 * np.sin(theta)**2*(x - x0)/sigma_y**2 
            dZdx0 = part1 - part2 - part3 

            # Z = X**2 + Y**2
            mdl = -A*np.exp(-X**2 - Y**2) * dZdx0 
            return mdl 
        def dy0(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  

            
            part1 = 2*(sigma_x - sigma_y)*(sigma_x + sigma_y) * np.sin(theta)*np.cos(theta)*(x-x0)/(sigma_x*sigma_y)**2 
            part2 = 2 * np.sin(theta)**2*(y - y0)/sigma_x**2 
            part3 = 2 * np.cos(theta)**2*(y - y0)/sigma_y**2 
            dZdx0 = part1 - part2 - part3 
            # Z = X**2 + Y**2
            mdl = -A*np.exp(-X**2 - Y**2) * dZdx0 
            return mdl 
        def dsigmax(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  

            dZdx0 = -2 * (np.cos(theta)*(x-x0) + np.sin(theta)*(y-y0))**2/sigma_x**3 

            # Z = X**2 + Y**2
            mdl = -A*np.exp(-X**2 - Y**2) * dZdx0 
            return mdl 
        def dsigmay(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  

            dZdx0 = -2 * (np.sin(theta)*(x-x0) + np.cos(theta)*(y-y0))**2/sigma_y**3 

            # Z = X**2 + Y**2
            mdl = -A*np.exp(-X**2 - Y**2) * dZdx0 
            return mdl 
        def dtheta(P0, source_map):
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  

            dZdx0 = -2*(sigma_x**2 - sigma_y**2) * (np.sin(theta)*dx + np.cos(theta)*dy) *\
                (np.cos(theta)*dx + np.sin(theta)*dy)/(sigma_x*sigma_y)**2

            # Z = X**2 + Y**2
            mdl = -A*np.exp(-X**2 - Y**2) * dZdx0 
            return mdl 
        
        def jacobian(P0:list, source_map : SkyMap):
            """Return a list of jacobian vectors"""
            
            j = np.array([j(P0, source_map) for j in [dA, dB, dx0, dy0, dsigmax, dsigmay, dtheta]]) 
            return j
        
        def model(P0:list, source_map : SkyMap):
            
            A, B, x0, y0, sigma_x, sigma_y, theta = P0 
            
            x,y = source_map.world_flat
            x[x > 180] -= 360
            dx = (x - x0)
            dy = (y - y0)
            

            X = dx*np.cos(theta)/sigma_x +\
                dy*np.sin(theta)/sigma_x
            Y = -dx*np.sin(theta)/sigma_y +\
                 dy*np.cos(theta)/sigma_y  
                 
            mdl = A*np.exp(-X**2 - Y**2) + B
            return mdl 
        
        def error(P0 : list, source_map : SkyMap):
            return np.sum((source_map.m_flat - model(P0, source_map))**2*source_map.weights_flat)
        
        
        beam = 4.5/60./2.355
        P0 = [np.max(source_map.m_flat), 0, 0, 0, beam, beam, 0]
        result = minimize(error, P0, args=(source_map),
                          bounds=[[0,None], #A 
                                  [None,None], #B
                                  [None,None], #x0 
                                  [None,None], #y0
                                  [0,None], # sigmax
                                  [0,None], # sigmay
                                  [-np.pi,np.pi]]) # theta
        
        J = jacobian(result.x, source_map) 
        C_data = np.diag(source_map.variance_flat)
        try:
            JJ_inv = np.linalg.inv(J.dot(J.T))
            C_parameters = JJ_inv.dot(J.dot(C_data.dot(J.T.dot(JJ_inv)))) 
        
            return result.x, np.diag(C_parameters)**0.5
        except np.linalg.LinAlgError: 
            return result.x, np.zeros(result.x.size)+np.nan

    
    def fit_source(self, data : HDF5Data, level2_data : COMAPLevel2):
        """ """
        
        self.source = data.source_name

        # Check source else return 
        if not self.source_good(): return
        
        self.source_position = SourcePosition(source=self.source,
                                              _mjd = level2_data.mjd) 
        
        n_feeds, n_bands, n_tod = level2_data.tod_shape 
        
        self.data[f'{self.calibration}_source_fit/fits'] = np.zeros((n_feeds,n_bands,self.NPARAMS, ))
        self.data[f'{self.calibration}_source_fit/errors'] = np.zeros((n_feeds,n_bands,self.NPARAMS, ))
        # Perform the fit for each feed band 
        for ((ifeed, feed),iband) in tqdm(level2_data.tod_loop(bands=True), desc='Fitting Sources'):
            tod = level2_data.tod[ifeed,iband,level2_data.on_source]
            tod_clean = tod - self.median_filter(tod, self.MEDIAN_FILTER_STEP)
            rms = level2_data.tod_auto_rms(ifeed, iband)
            weights = np.ones(tod_clean.size)/rms**2 
            

            self.source_position.set_mask(level2_data.on_source)
            relative_source_position = self.source_position.rotate_ra(level2_data.ra[ifeed,level2_data.on_source],
                                                                      level2_data.dec[ifeed,level2_data.on_source])
            
            source_map = SkyMap()
            source_map.set_map_info(self.NXPIX,self.NYPIX, 
                                    crval=[0,0],
                                    crpix=[self.NXPIX//2, self.NYPIX//2],
                                    ctype=['RA---TAN','DEC--TAN'],
                                    cdelt=[-1./60,1./60.]
                                    )
            source_map.bin_data(tod_clean,
                                relative_source_position.ra, 
                                relative_source_position.dec,
                                weights=weights)
                        
            result, errors = self.fit(source_map)  
            
            self.data[f'{self.calibration}_source_fit/fits'][ifeed,iband]  = result
            self.data[f'{self.calibration}_source_fit/errors'][ifeed,iband]  = errors
            logging.info(f'Feed {feed:02d} Band {iband:02d} A: {result[0]:.1f} pm {errors[0]:.1f}, x0: {result[2]:.1f} pm {errors[2]:.1f}')

        # return 
        
        