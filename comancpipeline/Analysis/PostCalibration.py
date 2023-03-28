#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:17 2023

Routines for applying the TauA/CasA/Jupiter calibration to the TODs

@author: sharper
"""
import numpy as np

from comancpipeline.Tools.CaliModels import TauAFluxModel 
from tqdm import tqdm 
import h5py 
import os

from .Running import PipelineFunction
from .DataHandling import COMAPLevel2
from dataclasses import dataclass, field 

@dataclass
class ApplyCalibration(PipelineFunction):
    name : str = 'ApplyCalibration'
    
    overwrite : bool = False
    
    calibrator_filelist : list = field(default_factory=list) 
    temp_calibrator_file : str = 'scripts/calibrator_temp_info.npy' 
    overwrite_calibrator_file : bool = False 
    calibrator_source : str = 'undefined' 
    def __post_init__(self):
        
        self.data = {}
        self.attrs = {} 
    @property 
    def save_data(self):
        """Use full path that will be saved into the HDF5 file"""
            
        return self.data, self.attrs

    @staticmethod
    def dA(P0):
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        
        x,y = x0, y0
        dx = (x - x0)
        dy = (y - y0)
        
        X = dx*np.cos(theta)/sigma_x +\
            dy*np.sin(theta)/sigma_x
        Y = -dx*np.sin(theta)/sigma_y +\
             dy*np.cos(theta)/sigma_y  
             
        mdl = np.exp(-X**2 - Y**2)
        return mdl 

    @staticmethod
    def dB(P0):
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        return 1 

    @staticmethod
    def dx0(P0):
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        x,y =x0,y0
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

    @staticmethod    
    def dy0(P0):
        
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        x,y = x0, y0
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
    @staticmethod
    def dsigmax(P0):
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        x,y = x0,y0
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
    
    @staticmethod
    def dsigmay(P0):
        
        A, B, x0, y0, sigma_x, sigma_y, theta = P0 
        x,y = x0,y0
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
    
    @staticmethod
    def dtheta(P0):
          A, B, x0, y0, sigma_x, sigma_y, theta = P0 
          x,y = x0,y0
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
    
    def flux_error(self, fits, errors): 
        
        diffs = [self.dA,self.dsigmax,self.dsigmay] 
        
        sigma_out = 0 
        for idiff, diff in enumerate(diffs):
            sigma_out += diff(fits)**2 * errors[idiff]**2 
              
        beam = 2*np.pi * fits[4]*fits[5]*(np.pi/180.)**2 
        pixel_size = (1./60.*np.pi/180. )**2 
        N = beam/pixel_size
        
        return sigma_out**0.5/np.sqrt(N)  
    
    def get_source_flux(self,frequency : float, 
                        fits : np.ndarray[float,float], 
                        errors : np.ndarray[float,float]):
        """Calculates the flux density given gaussian fit parameters""" 
        
        kb = 1.38e-23 
        nu = frequency * 1e9
        c  = 3e8 
        conv = 2*kb * (nu/c)**2 * 1e26 * (np.pi/180.)**2 
        flux = 2*np.pi*fits[:,0]*fits[:,4]*fits[:,5] * conv 
            
        flux_errs = np.array([self.flux_error(fits[i],errors[i,[0,4,5]]) if fits[i,4] !=0 else 0 for i in range(flux.size)]) 
            
        flux_errs *= conv 
        
        return flux, flux_errs 
    
    def get_source_geometric_radius(self, fits : np.ndarray[float,float], 
                                    errors : np.ndarray[float,float]):
        """Calculates the geometric radius given gaussian fit parameters""" 
        
        radius = np.sqrt(fits[:,4]**2 + fits[:,5]**2)
        radius_err = (fits[:,4]/radius)**2 * errors[:,4]**2 +\
            (fits[:,5]/radius)**2 * errors[:,5]**2
        radius_err = radius_err**0.5 
        
        return radius, radius_err
    
    
    def read_data(self, filelist, save_file, overwrite=False): 
        
        if os.path.exists(save_file) and not overwrite:
            return np.load(save_file, allow_pickle=True).flatten()[0] 
        
        taua_all = [] 
        err_all = [] 
    
        mjd_all = [] 
        for filename in tqdm(filelist): 
            
            data = h5py.File(filename,'r')
            if not 'comap' in data:
                continue 
            
            if not ('TauA_source_fit' in data): continue 
        
    
            feeds = data['spectrometer/feeds'][...]-1
            taua = np.zeros((20, 4, 7)) 
            taua[feeds] = data['TauA_source_fit']['fits'][...]
            err = np.zeros((20, 4, 7)) 
            err[feeds] = data['TauA_source_fit']['errors'][...]
    
            mjd = data['spectrometer/MJD'][0]
            taua_all += [taua]
            err_all += [err]
    
            mjd_all += [mjd]
            data.close()
    
        taua_all = np.array(taua_all) 
        mjd_all = np.array(mjd_all) 
        err_all = np.array(err_all) 
        output =  {'fits':taua_all, 'MJD':mjd_all, 'errors':err_all}
        np.save(save_file, output) 
        
        return output 
    
    def create_source_mask(self, flux, flux_err, radius, radius_err,
                           min_flux = 10,
                           max_flux=1000,
                           max_flux_err = 10, 
                           min_flux_err = 0.5,
                           max_geo_radius_diff=1e-3):
        """Calculate the mask for the bad TauA fits"""
        mask = (flux_err > max_flux_err) | ~np.isfinite(flux) | (flux< min_flux) | (flux > max_flux) |\
            ~np.isfinite(flux_err) |(flux_err < min_flux_err) 
            
        mean_size = np.nanmedian(radius[~mask])
        mask = mask | (np.abs(radius - mean_size) > max_geo_radius_diff)
    
        return mask 
    
    def calculate_calibration_factors(self, filelist, temp_save_file='scripts/taua_fits.npy',
                                      temp_file_overwrite=False):
        """Calculate the calibration factors for the COMAP feeds""" 
        data = self.read_data(filelist, temp_save_file, overwrite=temp_file_overwrite)
        flux_model = TauAFluxModel()
    
        cal_data = {} 
        for iband,frequency in zip([0,1,2,3],[27,29,31,33]):
            flux_feed1, flux_err_feed1 = self.get_source_flux(frequency, 
                                             data['fits'][:,0,iband,:],
                                             data['errors'][:,0,iband,:])
            radius_feed1, radius_err_feed1 = self.get_source_geometric_radius(data['fits'][:,0,iband,:],
                                                             data['errors'][:,0,iband,:])
            mask_feed1 = self.create_source_mask(flux_feed1, flux_err_feed1, radius_feed1, radius_err_feed1)
    
            for ifeed in range(20):
    
                flux, flux_err = self.get_source_flux(frequency, 
                                                 data['fits'][:,ifeed,iband,:],
                                                 data['errors'][:,ifeed,iband,:])
                radius, radius_err = self.get_source_geometric_radius(data['fits'][:,ifeed,iband,:],
                                                                 data['errors'][:,ifeed,iband,:])
                mask = self.create_source_mask(flux, flux_err, radius, radius_err,
                                          max_flux_err = 10)
    
                mask = mask | mask_feed1 
                
                cal_data[(ifeed,iband)] = {'MJD': data['MJD'][~mask],
                                       'cal_factors': flux[~mask]/flux_model(frequency, data['MJD'][~mask]),
                                       'cal_errors': flux_err[~mask]/flux_model(frequency, data['MJD'][~mask])}
                
    
        return cal_data 
    
    def assign_calibration_factors(self, level2_data : COMAPLevel2, cal_data : dict):
        """Assigns the nearest good calibration factor""" 
        
        level2_mjd = level2_data.mjd[0] 
        
        nfeeds, nbands, nsamples = level2_data.tod_shape 
        
        cal_factors = np.zeros((nfeeds, nbands))
        cal_errors = np.zeros((nfeeds, nbands))
        cal_mjd = np.zeros((nfeeds, nbands))

        for ifeed, feed in enumerate(level2_data.feeds):
            for iband in range(level2_data.nbands): 
                
                if len(cal_data[(feed-1,iband)]['MJD']) > 0:
                    idx = np.argmin(np.abs(level2_mjd - cal_data[(feed-1,iband)]['MJD'])) 
                    cal_factors[ifeed,iband] = cal_data[(feed-1,iband)]['cal_factors'][idx]
                    cal_errors[ifeed,iband] = cal_data[(feed-1,iband)]['cal_errors'][idx]
                    cal_mjd[ifeed,iband] = cal_data[(feed-1,iband)]['MJD'][idx]

        self.data['astro_calibration/cal_factors'] = cal_factors
        self.data['astro_calibration/cal_errors'] = cal_errors
        self.data['astro_calibration/cal_mjd'] = cal_mjd

        self.attrs['astro_calibration'] = {'calibrator':self.calibrator_source} 
        
    def __call__(self, level2_data : COMAPLevel2):
        """ """ 
        cal_data = self.calculate_calibration_factors(self.calibrator_filelist, 
                                                      self.temp_calibrator_file,
                                                      self.overwrite_calibrator_file) 

        self.assign_calibration_factors(level2_data, cal_data)
        
        return self.STATE 
if __name__ == "__main__": 
    
    pass    

