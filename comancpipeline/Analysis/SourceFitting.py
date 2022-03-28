import concurrent.futures

import numpy as np
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis import Calibration, Statistics
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs, binFuncs, stats,CaliModels, FileTools
from comancpipeline.data import Data
from scipy.optimize import fmin, leastsq, minimize
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS

from comancpipeline.Analysis import CreateLevel3
from statsmodels import robust
from tqdm import tqdm

from functools import partial
import copy 
from comancpipeline.Tools.median_filter import medfilt

import h5py

import emcee

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os
import shutil

__version__ = 'v5'

def filter_tod(data,feedtod,ifeed,level2='level2'):
    """
    Subtract the median filters and atmospheric model

    Returns also the mask defining the 'scans'
    """
    nBands, nChans,nSamples = feedtod.shape

    mask = np.zeros(nSamples,dtype=bool)

    medfilt_coefficient = data[f'{level2}/Statistics/filter_coefficients'][ifeed,...]
    atmos               = data[f'{level2}/Statistics/atmos'][ifeed,...]
    atmos_coefficient   = data[f'{level2}/Statistics/atmos_coefficients'][ifeed,...]
    scan_edges          = data[f'{level2}/Statistics/scan_edges'][...]

    az = data['level1/spectrometer/pixel_pointing/pixel_az'][ifeed,:]
    el = data['level1/spectrometer/pixel_pointing/pixel_el'][ifeed,:]
    for iscan,(start,end) in enumerate(scan_edges):
        median_filter   = data[f'{level2}/Statistics/FilterTod_Scan{iscan:02d}'][ifeed,...]
        mask[start:end] = True
        N = int((end-start))
        for iband in range(nBands):
            for ichan in range(nChans):

                mdl = Statistics.AtmosGroundModel(atmos[iband,iscan],az[start:end],el[start:end]) *\
                               atmos_coefficient[iband,ichan,iscan,0]
                mdl += median_filter[iband,:N] * medfilt_coefficient[iband,ichan,iscan,0]
                feedtod[iband,ichan,start:end] -= mdl
                feedtod[iband,ichan,start:end] -= np.nanmedian(feedtod[iband,ichan,start:end])

    return feedtod, mask
                

def plot_map_image(P,m,xy):
    source_model = Fitting.Gauss2dRot()

    Nx, Ny = 120,120
    dx, dy = 0.25/60.,0.25/60.
    #xy = np.meshgrid(((np.arange(Nx)+0.5)*dx - Nx*dx/2.),
    #                ((np.arange(Ny)+0.5)*dy - Ny*dy/2.))
    extent = [np.min(xy[0]),np.max(xy[0]),np.min(xy[1]),np.max(xy[1])]
    def downsample_map(m,x,y):
        
        nbins = 30
        binedges = (np.linspace(np.min(x),np.max(x),nbins+1),np.linspace(np.min(y),np.max(y),nbins+1))
        
        gd = np.isfinite(m)
        try:
            down_m = np.histogram2d(x[gd],y[gd], binedges,weights=m[gd])[0]/np.histogram2d(x[gd],y[gd], binedges)[0]
        except:
            down_m = np.zeros((nbins,nbins))*np.nan
        return down_m

        
    mdl = np.reshape(source_model(P,xy),(Ny,Nx))
    res = np.reshape(m,(Ny,Nx)) - mdl
    pyplot.subplot(2,2,1)        
    pyplot.imshow(downsample_map(np.reshape(m,(Ny,Nx)),xy[0],xy[1]).T,interpolation='none',extent=extent,origin='lower')
    pyplot.contour(xy[0],xy[1],mdl-P[-1],[0.1*P[0],0.5*P[0]],colors='k',linewidths=2,alpha=0.5)
    pyplot.plot(P[1],P[3],'kx')
    pyplot.grid()
    pyplot.title('Data')

    pyplot.subplot(2,2,2)
    pyplot.imshow(downsample_map(mdl,xy[0],xy[1]).T,interpolation='none',extent=extent,origin='lower')
    pyplot.plot(P[1],P[3],'kx')
    pyplot.grid()
    pyplot.title('Model')
     
    pyplot.subplot(2,2,3)
    pyplot.imshow(downsample_map(res,xy[0],xy[1]).T,interpolation='none',extent=extent ,origin='lower')
    pyplot.grid()
    pyplot.title('Residual')

    pyplot.show()

#from comancpipeline.Tools import alglib_optimize

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


import time
def doFit(todFit, _x, _y, sbc,bootstrap=False):
    """
    Fitting function within the fit routine, allows for try/excepts

    Separated off to allow for parallelisation 

    """
    sb, chan = sbc
    #todFit = todFitAll[int(sb),int(chan),:]
    nans = (np.isnan(todFit) == False)

    if (np.nansum(todFit) == 0):
        print('WARNING: All values were NaN')
        return 
        
    todFit,x,y = todFit[nans],_x[nans],_y[nans]

    if bootstrap:
        Pout, Perr = [], []
        niter = 200
        for i in range(niter):#tqdm(range(niter)):
            select = np.random.uniform(0,todFit.size,size=todFit.size).astype(int)
            P,Pe = fitproc(todFit[select],x[select],y[select])
            Pout += [P]
            Perr += [Pe]

        Pout = np.array(Pout)
        Perr = np.array(Perr)
        return np.mean(Pout,axis=0),np.mean(Perr,axis=0), chan, sb
    else:
        P,Pe = fitproc(todFit,x,y)
        return P, Pe, chan, sb
    


def fitproc(todFit,x,y):
    P0 = [np.max(todFit) - np.median(todFit), # amplitude
          0/60., # az offset
          0/60., # el offset
          9./60./2.355, # sigma
          np.median(todFit), # offset
          0., # az gradient
          0.] # el gradient


    fout = leastsq(Fitting.ErrorLstSq, P0,
                   #Dfun = Fitting.DFuncGaussRotPlaneLstSq,
                   full_output=True, 
                   maxfev = 100,
                   args=(Fitting.Gauss2dSymmetricPlane , #Fitting.Gauss2dRotPlane,
                         Fitting.Gauss2dSymmetricPlaneLimits,
                         x, y, todFit, 0,0))
    
                    
    #fout[0][5] = np.mod(fout[0][5], 2*np.pi)
    cov = fout[1]
    if isinstance(cov, type(None)):
        ferr = fout[0]*0.
    else:
        resid = np.std(todFit-Fitting.Gauss2d(fout[0], x, y,0,0))
        cov *= resid**2
        ferr = np.sqrt(np.diag(cov))
        
    # if fout[0][2] > fout[0][4]: # want x to be smaller than y
    #     _temp = fout[0][2]*1.
    #     fout[0][2] = fout[0][4]*1.
    #     fout[0][4] = _temp
    #     fout[0][5] = np.mod(fout[0][5] - np.pi/2., np.pi)
    # else:
    #     fout[0][5] = np.mod(fout[0][5], np.pi)            

    return fout[0], ferr
        

def makemap(d,x,y,ra0=0,dec0=0, cd=1./60., nxpix=600, nypix=600):
    """
    Quick binning routine for generating a map of single observation,
    useful for finding peaks if the pointing is really bad.

    d - 1d-numpy array of receiver data
    x - x-coordinate system per TOD
    y - y-coordinate system per TOD
    """

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
    return m,pixCens,w

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
    """
    Fisher matrix for 2D gaussian fit with (non-rotated) elliptical beam + offset
    """
    
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


class FitSource(BaseClasses.DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, feeds='all',
                 database = None,
                 output_dir=None,
                 lon=Coordinates.comap_longitude,
                 lat=Coordinates.comap_latitude,
                 prefix='AstroCal',
                 dx=1., dy=1., Nx=60,Ny=60,
                 allowed_sources= ['jupiter','TauA','CasA','CygA','mars'],
                 binwidth = 1,
                 level2='level2',
                 fwhm_prior = 'none',
                 fit_alt_scan = False,
                 use_leastsqs=False,
                 use_bootstrap=False,
                 figure_dir='figures',
                 make_figures=False,
                 fitfunc='Gauss2dRot',
                 **kwargs):
        """
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'FitSource'
        self.feeds_select = feeds
        
        self.dx = dx/Nx
        self.dy = dy/Ny
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.nfeeds_total = int(20)


        self.database   = database
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.make_figures = make_figures
        self.figure_dir=figure_dir
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)

        self.prefix = prefix
        self.fitfunc= Fitting.__dict__[fitfunc]()
        self.fitfunc_fixed= Fitting.__dict__[f'{fitfunc}_FixedPos']()

        self.lon = lon
        self.lat = lat 
        self.nodata = False
        
        self.allowed_sources = allowed_sources

        self.binwidth = binwidth

        self.level2=level2

        self.model = Fitting.Gauss2dRot_General(use_leastsqs=use_leastsqs,use_bootstrap=use_bootstrap)

        self.fit_alt_scan = fit_alt_scan

        self.models = {'jupiter':CaliModels.JupiterFlux,
                       'TauA': CaliModels.TauAFlux,
                       'CygA': CaliModels.CygAFlux,
                       'CasA': CaliModels.CasAFlux}

        # Beam modes allowed: 
        # 1) ModelFWHMPrior - Use James beam model fits as a prior
        # 2) ModelFWHMFixed - Use James beam models to fix the FWHM
        # 3) BootstrapPrior - Fit width using averaged data, use fitted widths as priors on other frequencies
        # 4) FullBootstrapPrior - Use FWHM fitted to Full data average
        # 5) NoPrior - Use no prior on FWHM
        self.fwhm_prior = fwhm_prior

        # FWHM models derived from James' beam models - generally too big?
        self.xfwhm = np.poly1d([ 5.22778336e-03, -3.76962352e-01,  1.11007533e+01])
        self.yfwhm = np.poly1d([ 6.07782238e-03, -4.26787057e-01,  1.18196903e+01])

        # Fitted fwhm's
        self.fitted_fwhm = {} #{feed:np.poly1d(fit) for feed,fit in Data.average_beam_widths.items()}


    def __str__(self):
        return 'Fitting source using {}'.format(self.fitfunc.__name__)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        fdir  = data.filename.split(fname)[0]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        comment = self.getComment(data)
        self.source = self.getSource(data)

        if self.checkAllowedSources(data, self.source, self.allowed_sources):
            return data

        if 'Sky nod' in comment:
            return data
        if 'test' in comment:
            return data
        if 'Engineering' in comment:
            return data

        if isinstance(self.output_dir, type(None)):
            self.output_dir = f'{fdir}/{self.prefix}'

        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)
        if os.path.exists(outfile) & (not self.overwrite):
            self.logger(f'{fname}:{self.name}: Source fits for {fname} already exists ({outfile}).')
            data = self.setReadWrite(data)
            self.linkfile(data)
            return data 


        data = self.setReadWrite(data)


        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing source fits to {outfile}.')

        if not isinstance(self.database,type(None)):
            self.write_database(data)
        if not self.nodata:
            self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def run(self,data):
        """
        """
        
        fname = data.filename.split('/')[-1]
        # Get data structures we need
        nHorns, nSBs, nChans, nSamples = data[f'{self.level2}/averaged_tod'].shape

        self.feeds, self.feedlist, self.feeddict = self.getFeeds(data,self.feeds_select)
        freq = data[f'{self.level2}/frequency'][...]
        freqwidth = np.abs(freq[0,0]-freq[0,1])*self.binwidth * 1e3

        spike_mask = self.getSpikeMask(data)
        sky_data_flag = ~Calibration.get_vane_flag(data['level1']) & spike_mask
        assert np.sum(sky_data_flag) > 0, 'Error: The calibration vane is closed for all of this observation?'

        # Make sure we are actually using a calibrator scan
        if self.source in Coordinates.CalibratorList:
            # Next bin into maps
            self.logger(f'{fname}:{self.name}: Creating maps with bin width {freqwidth:.1f}MHz.')

            mjd = data['level1/spectrometer/MJD'][...]
            all_maps = {'cel':np.empty((len(self.feeds), nSBs, nChans),dtype=object),
                        'az': np.empty((len(self.feeds), nSBs, nChans),dtype=object)}
            self.source_positions = {k:a for k,a in zip(['az','el','ra','dec']
                                                        ,Coordinates.sourcePosition(self.source, 
                                                                                    mjd, 
                                                                                    self.lon, 
                                                                                    self.lat))}
            self.source_positions['mean_el'] = np.mean(self.source_positions['el'])
            self.source_positions['mean_az'] = np.mean(self.source_positions['az'])
            self.source_positions['pa'] = Coordinates.pa(self.source_positions['ra'],
                                                         self.source_positions['dec'],mjd,
                                                         Coordinates.comap_longitude,
                                                         Coordinates.comap_latitude)*np.pi/180.

            pbar = tqdm(total = len(self.feeds)*nSBs*nChans,desc='Creating Maps')
            for i,(ifeed,feed) in enumerate(zip(self.feedlist, self.feeds)):
                feedtod = data[f'{self.level2}/averaged_tod'][ifeed,...]
                feedtod,mask = filter_tod(data,feedtod,ifeed)
                coords = self.get_coords(data,ifeed,mask & spike_mask)

                pyplot.plot(coords['sky_data_flag'])
                pyplot.show()
                for iband in range(nSBs):
                    for ichan in range(nChans):
                        cel_maps, az_maps = self.create_maps(data,
                                                             feedtod[iband,ichan,coords['sky_data_flag']],
                                                             mjd[coords['sky_data_flag']],
                                                             coords)
                        all_maps['cel'][i,iband,ichan] = cel_maps
                        all_maps['az'][i,iband,ichan]  = az_maps
                        pbar.update(1)
            pbar.close()

            self.logger(f'{fname}:{self.name}: Fitting global source offset')
            # First get the positions of the sources from the feed average
            self.avg_map_fits = np.empty(nSBs,dtype=object)
            for iband in range(nSBs):
                self.avg_map_fits[iband] = self.fit_source_position(data,all_maps['cel'][:,iband,:])
                
            self.logger(f'{fname}:{self.name}: Fitting source bands ({freqwidth:.1f}MHz).')
            # Finally, fit the data in the maps
            self.map_fits = self.fit_map(data,all_maps['cel'])

            print(data['level2'].keys())
            
            self.flux, self.gains = self.calculate_gains(data,self.map_fits, self.avg_map_fits)
        else:
            self.nodata = True
            return

    def calculate_gains(self,data, map_fits, avg_map_fits):
        """
        Convert the fitted brightness temperatures into gains
        """
        nFeeds,nBands,nChans,nParams = map_fits['Values'].shape
        frequencies = data[f'{self.level2}/frequency'][...]
        kb = 1.38064852e-23
        c  = 2.99792458e8
        scale = 2 * kb * (1e9/ c)**2 * 1e26

        source = self.getSource(data)
        self.flux = np.zeros((len(self.feeds),nBands, nChans))
        self.gain = np.zeros((len(self.feeds),nBands, nChans))

        for i,(ifeed,feed) in enumerate(zip(self.feedlist,self.feeds)):
            for iband in range(nBands):
                nu = frequencies[iband]
                sigx = avg_map_fits[iband]['Values'][2] 
                sigy = avg_map_fits[iband]['Values'][2]*avg_map_fits[iband]['Values'][4]
                amps = map_fits['Values'][i,iband,:,0]
                self.flux[i,iband,:] = 2*np.pi*amps*sigx*sigy*(np.pi/180.)**2 * scale*nu**2
                mdl_flux = self.models[source](nu,map_fits['MJD'],return_jansky=True,allpos=True)
                self.gain[i,iband,:] = self.flux[i,iband,:]/mdl_flux

        return self.flux, self.gain


    def get_point_data(self,data,special_idx):
        """
        Get the ra/dec and az/el position of the on source stare
        """
        if len(special_idx) == 0:
            return {'ra':np.nan,
                    'dec':np.nan,
                    'az':np.nan,
                    'el':np.nan}

        top = special_idx[0]
        ra_point, dec_point = Coordinates.h2e_full(data['spectrometer/pixel_pointing/pixel_az'][0,:top],
                                                   data['spectrometer/pixel_pointing/pixel_el'][0,:top],
                                                   data['spectrometer/MJD'][:top],
                                                   Coordinates.comap_longitude,
                                                   Coordinates.comap_latitude)
        point_data = {'ra':np.nanmean(ra_point),
                      'dec':np.nanmean(dec_point),
                      'az':np.nanmean(data['spectrometer/pixel_pointing/pixel_az'][0,:top]),
                      'el':np.nanmean(data['spectrometer/pixel_pointing/pixel_el'][0,:top])}
        return point_data

    def get_sky_data_flag(self,data):
        """
        Get flag of just the 'scan' data.
        """
        sky_data_flag = ~Calibration.get_vane_flag(data['level1']) 
        features = self.getFeatures(data)
        features = np.log10(features)/np.log10(2)
        sky_data_flag = sky_data_flag & np.isfinite(features) & (features != 16)

        return sky_data_flag

    def get_coords(self,data,ifeed,mask):
        """
        Get the az/el and ra/dec coordinates of the observation
        """
        sky_data_flag = mask
        
        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][ifeed,:]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][ifeed,:]
        ra  = data['level1/spectrometer/pixel_pointing/pixel_ra'][ifeed,:]
        dec  = data['level1/spectrometer/pixel_pointing/pixel_dec'][ifeed,:]

        N = az.shape[0]//2 * 2
        daz = np.gradient(az[:])*50.
        daz = daz[sky_data_flag]
        az = az[sky_data_flag]
        el = el[sky_data_flag]
        ra = ra[sky_data_flag]
        dec=dec[sky_data_flag]
        cw  = daz > 1e-2
        ccw = daz < 1e-2

        return {'az':az,
                'el':el,
                'ccw':ccw,
                'cw':cw,
                'ra':ra,
                'dec':dec,
                'sky_data_flag':sky_data_flag}

    def get_pixel_positions(self,_x, _y,x0,y0,pa=0,invertx=False):
        x,y =Coordinates.Rotate(_x, _y,
                                x0,y0 ,0)
        xr =  x*np.cos(pa) + y*np.sin(pa)
        yr = -x*np.sin(pa) + y*np.cos(pa)

        if invertx:
            xr *= -1

        pixels,pX,pY = self.getpixels(xr,yr,self.dx,self.dy,self.Nx,self.Ny)
        return pixels, pX, pY,xr ,yr

    def create_single_map(self,tod,x,y,x0,y0):
        """
        Bin tod into a correctly rotated map
        """
        maps = {'map':np.zeros((self.Nx*self.Ny)),
                'cov':np.zeros((self.Nx*self.Ny))}


        pixels,xp,yp,r_x, r_y = self.get_pixel_positions(x,y,x0,y0,0,invertx=True)
        mask = np.ones(pixels.size,dtype=int)

        mask[(pixels == -1) | np.isnan(tod) | np.isinf(tod)] = 0
        rms = stats.AutoRMS(tod)
        weights = {'map':tod.astype(np.float64)/rms**2,
                   'cov':np.ones(tod.size)/rms**2}
        for k in maps.keys():
            binFuncs.binValues(maps[k],
                               pixels,
                               weights=weights[k],mask=mask)
            maps[k] = np.reshape(maps[k],(self.Ny,self.Nx))
        return maps

    def create_maps(self,data,tod,mjd,coords):
        """
        Bin maps into instrument frame centred on source
        """
        features = np.log10(self.getFeatures(data))/np.log10(2)
        special_idx = np.where((features==16))[0]
        point_data = self.get_point_data(data,special_idx)

        cel_maps = self.create_single_map(tod,
                                          coords['ra'],
                                          coords['dec'],
                                          self.source_positions['ra'][coords['sky_data_flag']],
                                          self.source_positions['dec'][coords['sky_data_flag']])
        az_maps = self.create_single_map(tod,
                                         coords['az'],
                                         coords['el'],
                                         self.source_positions['az'][coords['sky_data_flag']],
                                         self.source_positions['el'][coords['sky_data_flag']])
        cel_maps= self.average_maps(cel_maps)
        az_maps = self.average_maps(az_maps)
        xygrid  = np.meshgrid((np.arange(self.Nx)+0.5)*self.dx - self.Nx*self.dx/2.,
                              (np.arange(self.Ny)+0.5)*self.dy - self.Ny*self.dy/2.)
        
        
        cel_maps['xygrid']=xygrid
        cel_maps['StareCoords']= {**point_data,'pa':np.nanmean(self.source_positions['pa'])}
        az_maps['xygrid']=xygrid
        az_maps['StareCoords'] = {**point_data,'pa':np.nanmean(self.source_positions['pa'])}
        return cel_maps,az_maps

    def get_fwhm_prior(self,freq,feed):
        """
        Returns the appropriate fwhm_priors
        """
        self.fitted_fwhm = {feed:np.poly1d(fit) for feed,fit in Data.average_beam_widths.items()}


        if (self.fwhm_prior == 'ModelFWHMPrior'):
            P0_priors={'sigx':{'mean':self.xfwhm(freq)/60./2.355,
                               'width':self.xfwhm(freq)/60./2.355/1e2}}
        elif (self.fwhm_prior == 'DataFWHMPrior'):
            P0_priors={'sigx':{'mean':self.fitted_fwhm[feed](1./freq)/60./2.355,
                               'width':self.fitted_fwhm[feed](1./freq)/60./2.355/1e2}}
        else:
            P0_priors = {}

        return P0_priors

    def fit_altscan_position(self,data,scan_maps):
        """
        Performs a full fit to the maps obtain the source positions 

        Recommended only for use on the band average data to reduce uncertainties
        """
        fname = data.filename.split('/')[-1]

        # We do Jupiter in the Az/El frame but celestial in sky frame
        if not 0 in self.feedlist:
            return 
        self.model.set_fixed(**{})

        def limfunc(P):
            A,x0,sigx,y0,sigy,phi,B = P
            if (sigx < 0) | (sigy < 0):
                return True
            if (phi < -np.pi/2.) | (phi >= np.pi/2.):
                return True
            return False

        self.alt_scan_parameters = self.model.get_param_names()
        self.alt_scan_fits ={'CW':{'Values':np.zeros((self.model.nparams)),
                                   'Errors':np.zeros((self.model.nparams)),
                                   'Chi2': np.zeros((2))},
                             'CCW':{'Values':np.zeros((self.model.nparams)),
                                    'Errors':np.zeros((self.model.nparams)),
                                    'Chi2': np.zeros(2)}}
        for key in ['CW','CCW']:
            m,c,x,y,P0 = self.prepare_maps(scan_maps[key]['map'],scan_maps[key]['cov'],scan_maps[key]['xygrid'])

            freq = 30
            P0_priors = self.get_fwhm_prior(freq,1)
            # Perform the least-sqaures fit
            try:
                result, error,samples,min_chi2,ddof = self.model(P0, (x,y), m, c,
                                                   P0_priors=P0_priors,return_array=True)
                self.alt_scan_fits[key]['Values'][:] = result
                self.alt_scan_fits[key]['Errors'][:] = error
                self.alt_scan_fits[key]['Chi2'][:] = min_chi2,ddof

            except ValueError as e:
                try:
                    self.logger(f'{fname}:emcee:{e}',error=e)
                except TypeError:
                    self.logger(f'{fname}:emcee:{e}')

                
    def prepare_maps(self,_m,_c,xygrid):
        """
        Prepare maps for fitting by flattening array, and removing bad values.
        Returns estimate of P0 too.
        """
        m = _m.flatten()
        c = _c.flatten()
        gd = np.isfinite(m)
        m = m[gd]
        c = c[gd]

        assert (len(m) > 0),'No good data in map'

        x,y =xygrid
        x,y = x.flatten()[gd],y.flatten()[gd]
        P0 = {'A':np.nanmax(m),
              'x0':x[np.argmax(m)],
              'sigx':2./60.,
              'y0':y[np.argmax(m)],
              'sigy_scale':1,
              'phi':0,
              'B':0}
        P0 = {k:v for k,v in P0.items() if not self.model.fixed[k]}
        return m,c,x,y,P0

    def create_average_feed_map(self,maps):
        """
        Average together an array of dictionaries to create a single map
        """

        avg_map = {'map':np.zeros((self.Ny,self.Nx)),
                   'cov':np.zeros((self.Ny,self.Nx))}
        
        nchans = maps.shape[-1]
        for i,(ifeed,feed) in enumerate(zip(self.feedlist,self.feeds)):
            if feed == 20:
                continue
            for ichan in range(nchans):
                tmp = maps[i,ichan]['map']/maps[i,ichan]['cov']
                cov = 1./maps[i,ichan]['cov']
                gd  = np.isfinite(tmp)
                avg_map['map'][gd] += tmp[gd]
                avg_map['cov'][gd] += cov[gd]
                
        avg_map = self.average_maps(avg_map)
        return avg_map

    def fit_source_position(self,data, maps):
        """
        Performs a full fit to the maps obtain the source positions 

        Recommended only for use on the band average data to reduce uncertainties
        """

        feed_avg = self.create_average_feed_map(maps)
        fname = data.filename.split('/')[-1]

        # We do Jupiter in the Az/El frame but celestial in sky frame
        def limfunc(P):
            A,x0,sigx,y0,sigy,phi,B = P
            if (sigx < 0) | (sigy < 0):
                return True
            if (phi < -np.pi/2.) | (phi >= np.pi/2.):
                return True
            return False
        self.model.set_fixed(**{})

        self.avg_map_parameters = self.model.get_param_names()

        avg_map_fits   = {'Values': np.zeros((self.model.nparams))*np.nan,
                          'Errors': np.zeros((self.model.nparams))*np.nan,
                          'Chi2': np.zeros((2))*np.nan}
        
        try:
            m,c,x,y,P0 = self.prepare_maps(feed_avg['map'],feed_avg['cov'],maps[0,0]['xygrid'])
        except AssertionError:
            return avg_map_fits
        P0_priors = {}

        # Perform the least-sqaures fit
        try:
            gd = (c != 0) & np.isfinite(m) & np.isfinite(c)

            result, error,samples, min_chi2, ddof = self.model(P0, (x[gd],y[gd]), m[gd], c[gd],
                                                               P0_priors=P0_priors,return_array=True)
            avg_map_fits['Values'][:] = result
            avg_map_fits['Errors'][:] = error
            avg_map_fits['Chi2'][:] = min_chi2, ddof
        except ValueError as e:
            try:
                self.logger(f'{fname}:emcee:{e}',error=e)
            except TypeError:
                self.logger(f'{fname}:emcee:{e}')
        return avg_map_fits
                
    def fit_peak_az_and_el(self,data):
        """
        Use the best fit source position fit to determine the absolute azimuth and elevation position
        """

        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][0,:]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][0,:]
        tod_model = self.model.func(self.avg_map_fits['Values'][:], (az,el))
        imax = np.argmax(tod_model)
        az_max = az[imax]
        el_max = el[imax]
        self.az_el_peak   = {'AZ_PEAK': np.array([az_max]),
                             'EL_PEAK': np.array([el_max])}

    def fit_map(self,data,maps):
        """
        This function fits for the source in each channel 
        """
        fname = data.filename.split('/')[-1]

        mjd0 = data['level1/spectrometer/MJD'][0]


        # If the source is Jupiter we will use the beam model
        self.model.set_fixed(**{'x0':True,'y0':True,'phi':True,'sigx':True,'sigy_scale':True})
        def limfunc(P):
            A,sigx,sigy,B = P
            if (sigx < 0) | (sigy < 0):
                return True
            return False

        self.map_parameters = self.model.get_param_names()
        
        nFeeds, nBands, nChans = maps.shape
        # Setup fit containers
        self.map_fits ={'Values': np.zeros((nFeeds,
                                            nBands,
                                            nChans,
                                            self.model.nparams)),
                        'Errors': np.zeros((nFeeds,
                                            nBands,
                                            nChans,
                                            self.model.nparams)),
                        'Chi2': np.zeros((nFeeds,
                                          nBands,
                                          nChans,
                                          2)),
                        'MJD':mjd0}


        self.free_parameters = ['A','B']
        self.fixed_parameters = ['x0','sigx','y0','sigy_scale','phi']
        pbar = tqdm(total=nFeeds*nBands*nChans,desc=f'{self.name}:fit_map:{self.source}')
        for ifeed in self.feedlist:        
            for isb in range(nBands):
                for ichan in range(nChans):
                    try:
                        m,c,x,y,P0 = self.prepare_maps(maps[ifeed,isb,ichan]['map'],
                                                       maps[ifeed,isb,ichan]['cov'],
                                                       maps[ifeed,isb,ichan]['xygrid'])
                    except AssertionError:
                        pbar.update(1)
                        continue

                    if np.nansum(m) == 0:
                        pbar.update(1)
                        continue

                    P0_priors = {}

                    self.model.set_defaults(x0  =self.avg_map_fits[isb]['Values'][1],
                                            sigx=self.avg_map_fits[isb]['Values'][2],
                                            y0  =self.avg_map_fits[isb]['Values'][3],
                                            sigy_scale=self.avg_map_fits[isb]['Values'][4],
                                            phi =self.avg_map_fits[isb]['Values'][5])

                    try:
                        gd = (c != 0) & np.isfinite(m) & np.isfinite(c)

                        result, error,samples, min_chi2, ddof = self.model(P0, (x[gd],y[gd]), m[gd], c[gd],
                                                                           P0_priors=P0_priors,return_array=True)

                        self.map_fits['Values'][ifeed,isb,ichan,:] = result
                        self.map_fits['Errors'][ifeed,isb,ichan,:] = error
                        self.map_fits['Chi2'][ifeed,isb,ichan,:]   = min_chi2, ddof
                        
                    except ValueError as e:
                        pbar.update(1)
                        result = 0
                        error = 0
                        try:
                            self.logger(f'{fname}:emcee:{e}',error=e)
                        except TypeError:
                            self.logger(f'{fname}:emcee:{e}')

                    pbar.update(1)
        pbar.close()
        return self.map_fits

    def aperture_phot(self,data,x,y,v):
        """
        Get the integrated flux of source
        """
        r = np.sqrt((x-self.avg_map_fits['Values'][1])**2 + (y-self.avg_map_fits['Values'][3])**2)
        
        inner = (r < 8./60.) & np.isfinite(data) 
        outer = (r > 8.5/60.) & (r < 12./60.) & np.isfinite(data)

        annu = np.nanmedian(data[outer])
        annu_rms = np.nanstd(data[outer])
        flux = np.sum(data[inner]) - annu*np.sum(inner)

        c = 3e8
        kb=1.38e-23
        beam = (1./60.*np.pi/180.)**2
        factor = 2*kb*(v*1e9/c)**2 * beam * 1e26
        return flux*factor, annu_rms*np.sqrt(np.sum(inner))*factor
        

    def average_maps(self,maps):
        """
        Average the maps assuming dictionary: {'map':m*w,'cov':w}
        """
        maps['map'] = maps['map']/maps['cov']
        maps['cov'] = 1./maps['cov']
        return maps

    def getpixels(self,x,y,dx,dy,Nx,Ny):
        """
        Get pixels for a simple cartesian plate carree projection.
        """
        
        Dx = (Nx*dx)
        Dy = (Ny*dy)

        # Not Nx + 1 to account for rounding
        pX = (x/dx + (Nx + 2)/2.).astype(int)
        pY = (y/dy + (Ny + 2)/2.).astype(int)
        pixels = pX + pY*Nx
        pixels[((pX < 0) | (pX >= Nx)) | ((pY < 0) | (pY >= Ny))] = -1

        # here we do use Nx + 1 as you want the precise float value of the pixel.
        return pixels,x/dx + (Nx + 1)/2.,  y/dx + (Nx + 1.)/2.
        
        

    def filter_data(self,tod,sel,medfilt_size):
        """
        Generate an aggressive median filter to remove large-scale correlated noise.
        """
        
        filters = np.zeros((tod.shape[0],tod.shape[1],tod.shape[2],int(np.sum(sel))))
        for ifeed in tqdm(self.feedlist,desc=f'{self.name}:filters:{self.source}'):
            feed_tod = tod[ifeed,...] 
            for isb in range(tod.shape[1]):
                for ichan in range(tod.shape[2]):
                    z = feed_tod[isb,ichan,sel]
                    bad = np.where(np.isnan(z))[0]
                    if len(bad) == len(z):
                        continue
                    if len(bad) > 0:
                        good = np.where(np.isfinite(z))[0]
                        
                        nearest = [good[np.argmin(np.abs(good-b))] for b in bad]
                        z[bad] = z[nearest]
                    filters[ifeed,isb,ichan,:] = median_filter(z,medfilt_size)
                    
        return filters
                    
    def write_database(self,data):
        """
        Write the fits to the general database
        """

        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        frequency = data[f'{self.level2}/frequency'][...]
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)

        if self.name in grp:
            del grp[self.name]
        stats = grp.create_group(self.name)

        stats.attrs['fixed_parameters'] = self.fixed_parameters
        stats.attrs['free_parameters']  = self.free_parameters
        stats.attrs['source'] = self.getSource(data)

        
        nBands = self.avg_map_fits.shape[0]
        dnames = ['feeds','frequency','Fluxes','Gains','Values','Errors','Chi2','Az','El','MJD'] 
        dsets  = [self.feeds,frequency,self.flux, self.gain, 
                  self.map_fits['Values'],
                  self.map_fits['Errors'],
                  self.map_fits['Chi2'],
                  np.array([self.source_positions['mean_az']]), 
                  np.array([self.source_positions['mean_el']]),
                  self.map_fits['MJD']]
        for i in range(nBands):
            if isinstance(self.avg_map_fits[i],type(None)):
                continue
            dnames += [f'Avg_Values_Band{i}',f'Avg_Errors_Band{i}']
            dsets  += [self.avg_map_fits[i]['Values'],self.avg_map_fits[i]['Errors']]

        for (dname, dset) in zip(dnames, dsets):
            if dname in stats:
                del stats[dname]
            stats.create_dataset(dname,  data=dset)
        output.close()
        
        
    def write(self,data):
        """
        Write the Tsys, Gain and RMS to a pandas data frame for each hdf5 file.
        """        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # We will store these in a separate file and link them to the level2s
        fname = data.filename.split('/')[-1]
        units = {'A':'K','x0':'degrees','y0':'degrees','sigx':'degrees','sigy':'degrees','sigy_scale':'none','B':'K','phi':'radians'}

        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)

        print ('WRITING: ',outfile)
        output = h5py.File(outfile,'a')

        # Set permissions and group
        os.chmod(outfile,0o664)
        shutil.chown(outfile, group='comap')

        ##
        ## Narrow channel fits
        ##

        for valerr in ['Values','Errors','Chi2']:
            if f'Gauss_Narrow_{valerr}' in output:
                del output[f'Gauss_Narrow_{valerr}']
            gauss_fits = output.create_group(f'Gauss_Narrow_{valerr}')
            gauss_fits.attrs['FitFunc'] = self.model.__name__
            gauss_fits.attrs['source_el'] = self.source_positions['mean_el']
            gauss_fits.attrs['source_az'] = self.source_positions['mean_az']

            dnames = self.map_parameters
            dsets = [self.map_fits[valerr][...,iparam] for iparam in range(self.map_fits[valerr].shape[-1])]

            for (dname, dset) in zip(dnames, dsets):
                if dname in output:
                    del output[dname]
                print(dname,dset.shape,units[dname])
                gauss_dset = gauss_fits.create_dataset(dname,  data=dset)
                gauss_dset.attrs['Unit'] = units[dname]
        

        output.attrs['SourceFittingVersion'] = __version__
        output.attrs['source'] = self.getSource(data)
        output.close()
        self.linkfile(data)

    def linkfile(self,data):
        fname = data.filename.split('/')[-1]
        lvl2 = data[self.level2]
        if self.prefix in lvl2:
            del lvl2[self.prefix]
        lvl2[self.prefix] = h5py.ExternalLink('{}/{}_{}'.format(self.output_dir,self.prefix,fname),'/')




    
