#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:47:17 2023

Routines for applying the TauA/CasA/Jupiter calibration to the TODs

@author: sharper
"""
import numpy as np

from comancpipeline.Tools.CaliModels import TauAFluxModel, JupiterFluxModel, CasAFluxModel
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
    calibrator_models : dict = field(default_factory=lambda:{'TauA':TauAFluxModel(), 'CasA':CasAFluxModel(), 'jupiter':JupiterFluxModel()})
    figure_directory : str = 'figures'
    nowrite : bool = False 

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
    
    def flux_error(self, fits, error, frequency): 
        
        diffs = [self.dA,self.dsigmax,self.dsigmay] 
        kb = 1.38e-23 
        c  = 2.99792458e8
        conv = 2*kb * (frequency/c)**2 * 1e26 * (np.pi/180.)**2    
        beam = 2*np.pi * fits[1]*fits[2]*conv
        flux = fits[0] * beam 
        sigma_out =  flux**2 * ((error[0]/fits[0])**2 +(error[1]/fits[1])**2 + (error[2]/fits[2])**2)

        return sigma_out**0.5
    
    def get_source_flux(self,frequency : float, 
                        fits : np.ndarray[float,float], 
                        errors : np.ndarray[float,float]):
        """Calculates the flux density given gaussian fit parameters""" 
        
        kb = 1.38e-23 
        nu = frequency * 1e9
        c  = 2.99792458e8
        conv = 2*kb * (nu/c)**2 * 1e26 * (np.pi/180.)**2 
        if fits.ndim == 1:
            flux = fits[0] * 2*np.pi*fits[4]*fits[5] * conv
            flux_errs = self.flux_error(fits[[0,4,5]],errors[[0,4,5]], nu)
        else:
            flux = 2*np.pi*fits[:,0]*fits[:,4]*fits[:,5] * conv 
            print('AMPLITUDE', nu*1e-9, np.nanmedian(fits[:,0]))
            print('FLUX',np.nanmedian(flux))
            print('BEAM AREA', np.nanmedian(2*np.pi*fits[:,4]*fits[:,5]) * (np.pi/180.)**2)
            print('WIDTHS', np.nanmedian(fits[:,4])*60*2.355, np.nanmedian(fits[:,5])*60*2.355)
            flux_errs = np.array([self.flux_error(fits[i,[0,4,5]],errors[i,[0,4,5]], nu) if fits[i,4] !=0 else 0 for i in range(flux.size)]) 
                    
        return flux, flux_errs 
    
    def get_source_geometric_radius(self, fits : np.ndarray[float,float], 
                                    errors : np.ndarray[float,float]):
        """Calculates the geometric radius given gaussian fit parameters""" 
        
        if fits.ndim==1:
            radius = np.sqrt(fits[4]**2 + fits[5]**2)
            radius_err = (fits[4]/radius)**2 * errors[4]**2 +\
                (fits[5]/radius)**2 * errors[5]**2
        else:
            radius = np.sqrt(fits[:,4]**2 + fits[:,5]**2)
            radius_err = (fits[:,4]/radius)**2 * errors[:,4]**2 +\
                (fits[:,5]/radius)**2 * errors[:,5]**2
        radius_err = radius_err**0.5 
        
        return radius, radius_err
    
    def get_source_mask(self, ifeed, h, mjd, frequency=27e9,iband=0):
        calibrator_source = h['comap'].attrs['source'].split(',')[0] #.decode('utf-8')
        data = h[f'{calibrator_source}_source_fit']
        flux_model = self.calibrator_models[calibrator_source]

        flux_feed1, flux_err_feed1 = self.get_source_flux(frequency, 
                                            data['fits'][ifeed,iband,:],
                                            data['errors'][ifeed,iband,:])
        radius_feed1, radius_err_feed1 = self.get_source_geometric_radius(data['fits'][ifeed,iband,:],
                                                            data['errors'][ifeed,iband,:])
        
        mask_feed1 = self.create_source_mask(flux_feed1, flux_err_feed1, radius_feed1, radius_err_feed1, flux_feed1/flux_model(frequency, mjd))

        return mask_feed1
    
    def read_data(self, filelist, save_file, overwrite=False): 
        
        if os.path.exists(save_file) and not overwrite:
            return np.load(save_file, allow_pickle=True).flatten()[0], None
        
        src_all = [] 
        err_all = [] 
        mjd_all = [] 
        el_all = []
        if len(filelist) == 0:
            raise FileExistsError("Filelist is empty")
        
        for filename in tqdm(filelist): 
            
            data = h5py.File(filename,'r')
            if not 'comap' in data:
                continue 
            
            if not (f'{self.calibrator_source}_source_fit' in data): continue 
        
    
            feeds = data['spectrometer/feeds'][...]-1
            src = np.zeros((20, 4, 7)) 
            src[feeds] = data[f'{self.calibrator_source}_source_fit']['fits'][...]
            err = np.zeros((20, 4, 7)) 
            err[feeds] = data[f'{self.calibrator_source}_source_fit']['errors'][...]
            el  = np.zeros((20)) 
            el[feeds] = np.nanmedian(data['spectrometer/pixel_pointing/pixel_el'][...],axis=-1)
            mjd = data['spectrometer/MJD'][0]
            src_all += [src]
            err_all += [err]
            el_all += [el]
    
            mjd_all += [mjd]
            data.close()
    
        src_all = np.array(src_all) 
        mjd_all = np.array(mjd_all) 
        err_all = np.array(err_all) 
        el_all  = np.array(el_all)
        data =  {'fits':src_all, 'MJD':mjd_all, 'errors':err_all, 'el':el_all}

        flux_model = self.calibrator_models[self.calibrator_source]
        cal_data = {} 
        unmasked_cal_data = {}
        for iband,frequency in zip([0,1,2,3],[27,29,31,33]):
            flux_feed1, flux_err_feed1 = self.get_source_flux(frequency, 
                                             data['fits'][:,0,iband,:],
                                             data['errors'][:,0,iband,:])
            radius_feed1, radius_err_feed1 = self.get_source_geometric_radius(data['fits'][:,0,iband,:],
                                                             data['errors'][:,0,iband,:])
            mask_feed1 = self.create_source_mask(flux_feed1, flux_err_feed1, radius_feed1, radius_err_feed1, flux_feed1/flux_model(frequency, data['MJD'][:]))
    
            #mask_flux = flux_feed1[~mask_feed1]/flux_model(frequency, data['MJD'][~mask_feed1])
            #mask_filelist = np.array(filelist)[~mask_feed1] 
            #idx = np.argsort(mask_flux)
            
            #print(mask_flux[idx[:10]])
            #print(np.sort(mask_filelist[idx[:10]]))
            for ifeed in range(20):
    
                flux, flux_err = self.get_source_flux(frequency, 
                                                 data['fits'][:,ifeed,iband,:],
                                                 data['errors'][:,ifeed,iband,:])
                radius, radius_err = self.get_source_geometric_radius(data['fits'][:,ifeed,iband,:],
                                                                 data['errors'][:,ifeed,iband,:])
                mask = self.create_source_mask(flux, flux_err, radius, radius_err,flux/flux_model(frequency, data['MJD'][:]),
                                          max_flux_err = 2)
    
                mask = mask | mask_feed1 
                
                #print('FLUX MODEL',np.nanmedian(flux_model(frequency, data['MJD'])),np.nanmedian(flux), np.nanmedian(flux_err), np.nanmedian(flux)/np.nanmedian(flux_model(frequency, data['MJD'])))

                cal_data[(ifeed,iband)] = {'MJD': data['MJD'][~mask],
                                           'EL': data['el'][~mask,ifeed],
                                       'cal_factors': flux[~mask]/flux_model(frequency, data['MJD'][~mask]),
                                       'cal_errors': flux_err[~mask]/flux_model(frequency, data['MJD'][~mask])}
                unmasked_cal_data[(ifeed,iband)] = {'MJD': data['MJD'],
                                           'EL': data['el'][~mask,ifeed],
                                       'cal_factors': flux/flux_model(frequency, data['MJD']),
                                       'cal_errors': flux_err/flux_model(frequency, data['MJD'])}

        np.save(save_file, cal_data) 

        return cal_data,unmasked_cal_data
    
    def create_source_mask(self, flux, flux_err, radius, radius_err, cali_factor, 
                           max_cali_factor=1,
                           min_cali_factor=0.5,
                           min_flux = 10,
                           max_flux=1000,
                           max_flux_err = 10, 
                           min_flux_err = 0.5,
                           max_geo_radius_diff=1e-3):
        """Calculate the mask for the bad SOURCE fits"""
        mask = (flux_err > max_flux_err) | ~np.isfinite(flux) |\
                ~np.isfinite(flux_err) |\
                (cali_factor > max_cali_factor) |\
                (cali_factor < min_cali_factor)
            
        #mean_size = np.nanmedian(radius[~mask])
        #mask = mask | (np.abs(radius - mean_size) > max_geo_radius_diff) # (flux< min_flux) | (flux > max_flux) |\ # (flux_err < min_flux_err) |\
    
        return mask 
    
    def calculate_calibration_factors(self, filelist, temp_save_file=None,
                                      temp_file_overwrite=False):
        """Calculate the calibration factors for the COMAP feeds""" 
        if isinstance(temp_save_file, type(None)):
            temp_save_file = f'scripts/{self.calibrator_source}_fits.npy'

        import time 
        t0 = time.time()
        cal_data,unmasked_cal_data = self.read_data(filelist, temp_save_file, overwrite=temp_file_overwrite)

        t1 = time.time() 
        t2 = time.time()
        print(f'Time to read data: {t1-t0} s')
        print(f'Time to calculate calibration factors: {t2-t1} s')
        if temp_file_overwrite:
            from matplotlib import pyplot
            from astropy.time import Time
            for ifeed in range(19):
                for iband in range(4):
                    date = Time(unmasked_cal_data[(ifeed,iband)]['MJD'], format='mjd')
                    pyplot.errorbar(date.datetime, unmasked_cal_data[(ifeed,iband)]['cal_factors'],fmt='.',yerr=unmasked_cal_data[(ifeed,iband)]['cal_errors'])
                    date = Time(cal_data[(ifeed,iband)]['MJD'], format='mjd')
                    pyplot.errorbar(date.datetime, cal_data[(ifeed,iband)]['cal_factors'],fmt='.',yerr=cal_data[(ifeed,iband)]['cal_errors'])

                    pyplot.xlabel('Date')
                    pyplot.ylabel('Calibration Factor')
                    pyplot.title(f'Feed {ifeed+1}, Band {iband+1}')
                    pyplot.gcf().autofmt_xdate()
                    pyplot.ylim(0,2)
                    pyplot.xlim(Time('2019-01-01').datetime, Time('2023-01-01').datetime)
                    pyplot.grid()
                    pyplot.savefig(f'{self.figure_directory}/cal_factors_{self.calibrator_source}_feed{ifeed+1}_band{iband+1}.png')
                    pyplot.close()

            # Plot Elevation dependence 
            for ifeed in range(19):
                for iband in range(4):
                    pyplot.errorbar(cal_data[(ifeed,iband)]['EL'], cal_data[(ifeed,iband)]['cal_factors'],fmt='.',yerr=cal_data[(ifeed,iband)]['cal_errors'])

                    pyplot.xlabel('Elevation [degrees]')
                    pyplot.ylabel('Calibration Factor')
                    pyplot.title(f'Feed {ifeed+1}, Band {iband+1}')
                    pyplot.ylim(0.5,1)
                    pyplot.xlim(0,90)
                    pyplot.grid()
                    pyplot.savefig(f'{self.figure_directory}/cal_factors_vs_elevation_{self.calibrator_source}_feed{ifeed+1}_band{iband+1}.png')
                    pyplot.close()

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

        self.data[f'astro_calibration/{self.calibrator_source}_cal_factors'] = cal_factors
        self.data[f'astro_calibration/{self.calibrator_source}_cal_errors'] = cal_errors
        self.data[f'astro_calibration/{self.calibrator_source}_cal_mjd'] = cal_mjd
        
    def __call__(self, level2_data : COMAPLevel2):
        """ """ 
        cal_data = self.calculate_calibration_factors(self.calibrator_filelist, 
                                                      self.temp_calibrator_file,
                                                      self.overwrite_calibrator_file) 
        if not self.nowrite:
            print('hello')
            self.assign_calibration_factors(level2_data, cal_data)
        return self.STATE 
if __name__ == "__main__": 
    
    pass    

