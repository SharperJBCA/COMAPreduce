import concurrent.futures

import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs, binFuncs, stats
from scipy.optimize import fmin, leastsq, minimize
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust
import pandas as pd
from tqdm import tqdm

from functools import partial
import copy 
from comancpipeline.Tools.median_filter import medfilt

import h5py

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os
import shutil

__version__ = 'v2'

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
        #import corner
        #corner.corner(Pout)
        #pyplot.show()
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


class FitSource(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, feeds='all',
                 output_dir='AstroCal',
                 lon=-118.2941, lat=37.2314,
                 prefix='AstroCal',
                 dx=1., dy=1., Nx=60,Ny=60,
                 fitfunc='Gauss2dRot'):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """

        self.feeds = feeds
        
        self.dx = dx/60
        self.dy = dy/60.
        self.Nx = int(60)
        self.Ny = int(60)
        self.nfeeds_total = int(20)

        self.output_dir = output_dir
        self.prefix = prefix
        self.fitfunc= Fitting.__dict__[fitfunc]()
        self.fitfunc_fixed= Fitting.__dict__[f'{fitfunc}_FixedPos']()

        self.lon = lon
        self.lat = lat 
        self.nodata = False

    def __str__(self):
        return 'Fitting source using {}'.format(self.fitfunc.__name__)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        allowed_sources = ['jupiter','TauA','CasA','CygA','mars']
        
        self.source = data['level1/comap'].attrs['source']#.decode('utf-8').split(',')[0]
        if not isinstance(self.source,str):
            self.source = self.source.decode('utf-8').split(',')[0]
        else:
            self.source = self.source.split(',')[0]
        comment = data['level1/comap'].attrs['comment']#.decode('utf-8')
        if not isinstance(comment,str):
            comment = comment.decode('utf-8')

        if not self.source in allowed_sources:
            return data
        if 'Sky nod' in comment:
            return data
        if 'test' in comment:
            return data

        print('Starting Run')
        self.run(data)

        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        if not self.nodata:
            self.write(data)

        return data

    def run(self,data):
        # Get data structures we need
        alltod = data['level2/averaged_tod']

        if self.feeds == 'all':
            feeds = data['level1/spectrometer/feeds'][:]
        else:
            if not (isinstance(self.feeds,list)) | (isinstance(self.feeds,np.ndarray)):
                self.feeds = [int(self.feeds)]
                feeds = self.feeds
            else:
                feeds = [int(f) for f in self.feeds]
        self.feeds    = feeds
        self.feedlist = data['level1/spectrometer/feeds'][:]
        self.feeddict = {feedid:feedindex for feedindex, feedid in enumerate(self.feedlist)}
        #mjd = data['level1/spectrometer/MJD'][:]
        #az  = data['level1/spectrometer/pixel_pointing/pixel_az'][:]
        #el  = data['level1/spectrometer/pixel_pointing/pixel_el'][:]
        src = self.source
        freq = data['level2/frequency'][...]
        features = (np.log(data['level1/spectrometer/features'][:])/np.log(2)).astype(int)
        filename = data.filename


        # Make sure we are actually using a calibrator scan
        if self.source in Coordinates.CalibratorList:
            nHorns, nSBs, nChans, nSamples = alltod.shape   

            # First we will apply a median filter
            filters = self.filter_data(alltod,features,500)

            # Next bin into maps
            print('Creating Maps')
            self.maps, self.covs, xygrid, self.feed_avg_maps, self.feed_avg_covs = self.create_maps(data,alltod,filters,features)#,mjd)
            
            print('Fitting Sources')
            # First get the positions of the sources from the feed average
            self.fit_source_position(self.feed_avg_maps, self.feed_avg_covs,xygrid)

            # Finally, fit the data in the maps
            self.fit_map(self.maps,self.covs,xygrid,freq)
        else:
            self.nodata = True
            return


    def fit_source_position(self, maps, covs, xygrid):
        """
        Performs a full fit to the maps obtain the source positions 

        Recommended only for use on the band average data to reduce uncertainties
        """
        def limfunc(P):
            return False

        # Unpack the pixel coordinates
        x,y = xygrid
        x = x.flatten()
        y = y.flatten()

        # Store the feed average fits
        nparams = 7
        self.feed_avg_fits = np.zeros((maps.shape[0],nparams))
        self.feed_avg_uncertainty = np.zeros((maps.shape[0],nparams)) 
        for ifeed in tqdm(range(maps.shape[0]),desc=f'Fitting rank {rank}'):
            data = maps[ifeed,...].flatten()
            cov  = covs[ifeed,...].flatten()

            gd = np.isfinite(data)
            if np.sum(gd) == 0:
                continue

            xgd = x[gd]
            ygd = y[gd]

            # Give some start parameters
            P0 = [np.nanmax(data)-np.nanmedian(data),
                  xgd[np.argmax(data[gd])], # x0
                  1./60., #sigx
                  ygd[np.argmax(data[gd])], # y0
                  1./60., # sigy
                  0., # phi
                  np.nanmedian(data)] # bkgd

            # Perform the least-sqaures fit
            result = minimize(Fitting.ErrorLstSq,P0,args=[self.fitfunc.func,limfunc,[x[gd],y[gd]],data[gd],cov[gd],{}],method='CG')

            self.feed_avg_fits[ifeed,:] = result.x

    def fit_map(self,maps,covs,xygrid,freq):
        def limfunc(P):
            return False

        # Unpack Pixel Coordinates
        x,y = xygrid
        x = x.flatten()
        y = y.flatten()

        # Setup fit containers
        nparams = 4 # we fix the position and rotation
        self.fits          = np.zeros((maps.shape[0],maps.shape[1],maps.shape[2],nparams))
        self.fitsuccess    = np.zeros((maps.shape[0],maps.shape[1],maps.shape[2]),dtype=bool)
        self.uncertainties = np.zeros((maps.shape[0],maps.shape[1],maps.shape[2],nparams))
        self.apers         = np.zeros((maps.shape[0],maps.shape[1],maps.shape[2],2))

        for ifeed in tqdm(range(maps.shape[0]),desc=f'Fitting rank {rank}'):
            for isb in range(maps.shape[1]):
                for ichan in range(maps.shape[2]):
                    data = maps[ifeed,isb,ichan,...].flatten()
                    cov  = covs[ifeed,isb,ichan,...].flatten()
                    if np.nansum(data) == 0:
                        self.fitsuccess[ifeed,isb,ichan] = False
                        continue

                    self.apers[ifeed,isb,ichan,:] = self.aperture_phot(data,x,y,freq[isb,ichan])
                    gd = np.isfinite(data)
                    xgd = x[gd]
                    ygd = y[gd]
                    P0 = [np.nanmax(data)-np.nanmedian(data),
                          1./60., # sigx
                          1./60., # sigy
                          np.nanmedian(data)] # bkgd
                    fit_pos = {'x0':self.feed_avg_fits[ifeed,1], 
                               'y0':self.feed_avg_fits[ifeed,3],
                               'phi':self.feed_avg_fits[ifeed,5]}
                    args = [self.fitfunc_fixed.func,
                            limfunc,
                            [x[gd],y[gd]],
                            data[gd],
                            cov[gd],
                            fit_pos]
                    result = minimize(Fitting.ErrorLstSq,P0,args=args,method='CG')

                    # Calculate the uncertainties using the Jacobian of the fitting model
                    self.uncertainties[ifeed,isb,ichan,:] = self.fitfunc_fixed.covariance(result.x,
                                                                                          [x[gd],y[gd]],
                                                                                          data[gd], 
                                                                                          cov[gd]**0.5,
                                                                                          **fit_pos)

                    self.fits[ifeed,isb,ichan,:] = result.x
                    self.fitsuccess[ifeed,isb,ichan] = result.success

    def aperture_phot(self,data,x,y,v):
        """
        Get the integrated flux of source
        """
        r = np.sqrt(x**2 + y**2)
        
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
        
    def create_maps(self,data,tod,filters,features):
        """
        Bin maps into instrument frame centred on source
        """
        sel = np.where((features > 0) & (features != 13))[0]

        mjd = data['level1/spectrometer/MJD'][:]

        # We do Jupiter in the Az/El frame but celestial in sky frame
        if self.source.upper() == 'JUPITER':
            az  = data['level1/spectrometer/pixel_pointing/pixel_az'][:]
            el  = data['level1/spectrometer/pixel_pointing/pixel_el'][:]
            az = az[:,sel]
            el = el[:,sel]
        else:
            ra  = data['level1/spectrometer/pixel_pointing/pixel_ra'][:]
            dec = data['level1/spectrometer/pixel_pointing/pixel_dec'][:]
            ra  = ra[:,sel]
            dec = dec[:,sel]
        mjd=mjd[sel]

        npix = self.Nx*self.Ny

        maps = np.zeros((tod.shape[0],tod.shape[1],tod.shape[2],self.Nx,self.Ny))
        mapdata = np.zeros(npix,dtype=np.float64)
        hitdata = np.zeros(npix,dtype=np.float64)
        covs = np.zeros((tod.shape[0],tod.shape[1],tod.shape[2],self.Nx,self.Ny))

        feed_avg_map = np.zeros((tod.shape[0],self.Nx,self.Ny))
        feed_avg_cov = np.zeros((tod.shape[0],self.Nx,self.Ny))

        azSource, elSource, raSource, decSource = Coordinates.sourcePosition(self.source, mjd, self.lon, self.lat)
        for ifeed in tqdm(range(tod.shape[0])):
            feed_tod = tod[ifeed,...] 

            if self.source.upper() == 'JUPITER':
                x,y =Coordinates.Rotate(az[ifeed,:],
                                        el[ifeed,:],
                                        azSource, elSource,0)
            else:
                x,y =Coordinates.Rotate(ra[ifeed,:],
                                        dec[ifeed,:],
                                        raSource, decSource,0)


            
            pixels = self.getpixels(x,y,self.dx,self.dy,self.Nx,self.Ny)
            mask = np.ones(pixels.size,dtype=int)
            mask[pixels == -1] = 0
            for isb in range(tod.shape[1]):
                for ichan in range(tod.shape[2]):
                    mapdata[...] = 0.
                    hitdata[...] = 0.
                    z = (feed_tod[isb,ichan,sel]-filters[ifeed,isb,ichan])
                    if np.sum(np.isfinite(z)) == 0:
                        continue
                    rms = stats.AutoRMS(z)

                    binFuncs.binValues(mapdata, pixels, weights=z.astype(np.float64)/rms**2,mask=mask)
                    binFuncs.binValues(hitdata, pixels, mask=mask,weights=np.ones(z.size)/rms**2)

                    maps[ifeed,isb,ichan,...] = np.reshape(mapdata/hitdata,(self.Ny,self.Nx))
                    covs[ifeed,isb,ichan,...] = np.reshape(1./hitdata,(self.Ny,self.Nx))

                    feed_avg_map[ifeed,...] += np.reshape(mapdata,(self.Ny,self.Nx))
                    feed_avg_cov[ifeed,...] += np.reshape(hitdata,(self.Ny,self.Nx))

        xygrid = np.meshgrid(np.linspace(-self.Nx*self.dx/2.,self.Nx*self.dx/2.,self.Nx),
                             np.linspace(-self.Ny*self.dy/2.,self.Ny*self.dy/2.,self.Ny))
        
        feed_avg_map = feed_avg_map/feed_avg_cov
        feed_avg_cov = 1./feed_avg_cov

        return maps, covs, xygrid, feed_avg_map, feed_avg_cov

    def getpixels(self,x,y,dx,dy,Nx,Ny):
        
        Dx = (Nx*dx)
        Dy = (Ny*dy)

        pX = ((x+Dx/2.)/dx).astype(int)
        pY = ((y+Dy/2.)/dy).astype(int)
        pixels = pX + pY*Nx
        pixels[((pX < 0) | (pX >= Nx)) | ((pY < 0) | (pY >= Ny))] = -1

        return pixels
        
        

    def filter_data(self,tod,features,medfilt_size):
        sel = np.where((features > 0) & (features != 13))[0]

        
        filters = np.zeros((tod.shape[0],tod.shape[1],tod.shape[2],sel.size))
        for ifeed in tqdm(range(tod.shape[0])):
            feed_tod = tod[ifeed,...] 
            for isb in range(tod.shape[1]):
                for ichan in range(tod.shape[2]):
                    filters[ifeed,isb,ichan,:] = median_filter(feed_tod[isb,ichan,sel],medfilt_size)
                    
        return filters
                    

        
    def write(self,data):
        """
        Write the Tsys, Gain and RMS to a pandas data frame for each hdf5 file.
        """        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # We will store these in a separate file and link them to the level2s
        fname = data.filename.split('/')[-1]
        print('{}/{}_{}'.format(self.output_dir,self.prefix,fname))

        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)
        if os.path.exists(outfile):
            os.remove(outfile)

        output = h5py.File(outfile,'a')

        # Set permissions and group
        os.chmod(outfile,0o664)
        shutil.chown(outfile, group='comap')

        # Store datasets in root
        dnames = ['Aper_Amp','Aper_Err','Maps']
        dsets  = [self.apers[...,i] for i in range(self.apers.shape[-1])] +\
                 [self.maps]
                  
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)


        # Put the gaussian fit into its own groups
        gauss_fits = output.create_group('Gauss_Values')
        gauss_errs = output.create_group('Gauss_Errors')
        gauss_fits.attrs['FitFunc'] = self.fitfunc.__name__
        gauss_errs.attrs['FitFunc'] = self.fitfunc.__name__

        ####
        # Combine all the data into one data cube for output:
        ####
        all_fits = np.zeros((self.fits.shape[0], self.fits.shape[1], self.fits.shape[2], 7))
        all_errs = np.zeros((self.fits.shape[0], self.fits.shape[1], self.fits.shape[2], 7))
        for i,iparam in enumerate([1,3,5]):
            all_fits[:,:,:,iparam] = (self.feed_avg_fits[:,iparam])[:,None,None]
        for i,iparam in enumerate([0,2,4,6]):
            all_fits[:,:,:,iparam] = self.fits[...,i]
            all_errs[:,:,:,iparam] = self.uncertainties[...,i]

        dnames = ['Amp','x0','sigx','y0','sigy','angle','offset']
        units = {'Amp':'K','x0':'degrees','y0':'degrees','sigx':'degrees','sigy':'degrees','offset':'K','angle':'radians'}

        # Write best fit values
        dsets =  [all_fits[...,i] for i in range(all_fits.shape[-1])]
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            gauss_dset = gauss_fits.create_dataset(dname,  data=dset)
            gauss_dset.attrs['Unit'] = units[dname]

        gauss_fits.create_dataset('Success',data=self.fitsuccess)

        # Write uncertainties in best fit values
        dsets =  [all_errs[...,i] for i in range(all_errs.shape[-1])]
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            gauss_dset = gauss_errs.create_dataset(dname,  data=dset)
            gauss_dset.attrs['Unit'] = units[dname]

        output.attrs['SourceFittingVersion'] = __version__

        output['Aper_Amp'].attrs['Unit'] = 'Jy'
        output['Aper_Err'].attrs['Unit'] = 'Jy'

        output['Maps'].attrs['Unit'] = 'K'
        output['Maps'].attrs['cdeltx'] = self.dx
        output['Maps'].attrs['cdelty'] = self.dy
        
        output.attrs['Good'] = True
        output.close()
        fname = data.filename.split('/')[-1]
        lvl2 = data['level2']
        if self.prefix in lvl2:
            del lvl2[self.prefix]
        lvl2[self.prefix] = h5py.ExternalLink('{}/{}_{}'.format(self.output_dir,self.prefix,fname),'/')




    
class FitSourceAlternateScans(FitSource):
    def scanEdges(self,x, idir=0):
        """
        Calculate edges of raster scans.
        """
        assert idir < 2

        rms = np.std(x[1::2] - x[:-1:2]) /np.sqrt(2)
        cutdx = rms*3

        edges = ffuncs.scanedges(x, rms)
        edges = edges[edges != 0]
        diff = (edges[1:] - edges[:-1])
        bad = np.where(diff < 5)[0]
        edges = np.delete(edges, bad)

        scanEdges = edges.astype(int) #peaks[whePeaks]
        # d1 scans:
        select = np.zeros(x.size).astype(bool)
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

    def run(self,data):
        # Get data structures we need
        alltod = data['spectrometer/tod']
        btod = data['spectrometer/band_average']

        mjd = data['spectrometer/MJD'][:]
        az  = data['spectrometer/pixel_pointing/pixel_az'][:]
        el  = data['spectrometer/pixel_pointing/pixel_el'][:]
        src = self.source
        src = [s for s in src if s in Coordinates.CalibratorList]
        features = (np.log(data['spectrometer/features'][:])/np.log(2)).astype(int)
        filename = data.filename
        # Make sure we are actually using a calibrator scan
        if len(src) == 0:
            return
        self.source = src[0]

        self.outputs = {}
        if self.source in Coordinates.CalibratorList:
            tod = self.average(filename,data,alltod)

            cwbools = self.scanEdges(az[0,:], idir=0) # Get the CW scans (~cwbools is CCW scans)
            modes = {'CW':True,'CCW':False}
            for k,v in modes.items(): # loop twice for each direction             self.fit(data,cwbools==mode[i])
                self.fit(filename,tod[:,:,:,cwbools==v],
                         az[:,cwbools==v],
                         el[:,cwbools==v],
                         mjd[cwbools==v],
                         src,
                         features[cwbools==v])
                self.outputs[k] = [self.Pout,self.Perr,self.PeakAzEl, self.PeakRaDec]

    def write(self,data):
        """
        Write the Tsys, Gain and RMS to a pandas data frame for each hdf5 file.
        """        
        nHorns, nSBs, nChan, nSamps = data['spectrometer/tod'].shape
        nHorns = len(self.feeds)
        nChan = int(nChan//self.average_width) # need to capture the fact the data is averaged
        nParams = 9

        if not hasattr(self,'Pout'):
            return 
        # Structure:
        #                                    Frequency (GHz)
        # Date, DeltaT, Mode, Horn, SideBand     
        freq = data['spectrometer/frequency'][...]
        startDate = Types.Filename2DateTime(data.filename)

        # Reorder the data
        
        horns = self.feeds
        sidebands = np.arange(nSBs).astype(int)
        channels  = np.arange(nChan).astype(int)
        modes = ['Fits', 'Errors']
        direction = ['CW','CCW'] 
        iterables = [[startDate],[self.source],direction, modes, horns, sidebands, channels]
        names = ['Date','Source','Direction', 'Mode','Horn','Sideband','Channel']
        index = pd.MultiIndex.from_product(iterables, names=names)

        colnames = ['Amplitude','dAz','sig_az','dEl','sig_el','angle','offset','gradaz','gradel']
        addnames = ['Az','El']
        radecnames=['dRA','dDEC','MJD']
        df = pd.DataFrame(index=index, columns=colnames+addnames+radecnames)
        idx = pd.IndexSlice

        for k,v in self.outputs.items():
            Pout, Perr, PeakAzEl, PeakRaDec = v
            df.loc[idx[:,:,k,'Fits',:,:,:],colnames]   = np.reshape(Pout, (nHorns*nSBs*nChan,nParams))
            df.loc[idx[:,:,k,'Errors',:,:,:],colnames] = np.reshape(Perr, (nHorns*nSBs*nChan,nParams))
            df.loc[idx[:,:,k,'Fits',:,:,:],addnames]  = np.reshape(PeakAzEl, (nHorns*nSBs*nChan,2))
            df.loc[idx[:,:,k,'Fits',:,:,:],radecnames]  = np.reshape(PeakRaDec, (nHorns*nSBs*nChan,3))

        df = df.infer_objects() # convert to floats
        prefix = data.filename.split('/')[-1].split('.hd5')[0]
        df.to_pickle('{}/{}_SourceFitsAlternateScans.pkl'.format(self.output_dir,prefix))
