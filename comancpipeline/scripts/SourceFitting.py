from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis import Calibration
from comancpipeline.data import Data

from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs, binFuncs, stats
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from comancpipeline.Tools.median_filter import medfilt

import concurrent.futures
import numpy as np
import h5py
from tqdm import tqdm
from matplotlib import pyplot
from scipy.ndimage.filters import median_filter
import os

class FitSource(DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, 
                 feeds='all',
                 output_dir='',
                 lon=-118.2941, lat=37.2314,
                 prefix='AstroCal',
                 dx=1., dy=1., Nx=60,Ny=60,
                 allowed_sources= ['jupiter','TauA','CasA','CygA','mars'],
                 fit_alt_scan = True,
                 fitfunc='Gauss2dRot',**kwargs):
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

        self.output_dir = output_dir
        self.prefix = prefix
        self.fitfunc= Fitting.__dict__[fitfunc]()
        self.fitfunc_fixed= Fitting.__dict__[f'{fitfunc}_FixedPos']()

        self.lon = lon
        self.lat = lat 
        self.nodata = False
        
        self.allowed_sources = allowed_sources


        self.model = Fitting.Gauss2dRot_General()

        self.fit_alt_scan = fit_alt_scan

    def __str__(self):
        return 'Fitting source using {}'.format(self.fitfunc.__name__)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        fdir  = data.filename.split(fname)[0]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        comment     = self.getComment(data)
        self.source = self.getSource(data)

        print(comment,self.source)
        if self.checkAllowedSources(data, self.source, self.allowed_sources):
            return data

        if 'Sky nod' in comment:
            return data
        if 'test' in comment:
            return data

        if isinstance(self.output_dir, type(None)):
            self.output_dir = f'{fdir}/{self.prefix}'

        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)

        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing source fits to {outfile}.')
        if not self.nodata:
            self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def run(self,data):
        """
        """

        fname = data.filename.split('/')[-1]
        # Get data structures we need
        tod             = data[f'spectrometer/band_average']
        #calvane_factors = ScriptTools.QuickCalibration(tod,data) # Do a quick Cal vane calibration
        freq     = np.mean(data[f'spectrometer/frequency'][...],axis=-1)
        self.feeds, self.feedlist, self.feeddict = self.getFeeds(data,self.feeds_select)

        
        sky_data_flag = ~Calibration.get_vane_flag(data) 
        assert np.sum(sky_data_flag) > 0, 'Error: The calibration vane is closed for all of this observation?'

        # Make sure we are actually using a calibrator scan
        if self.source in Coordinates.CalibratorList:
            
            # First we will apply a median filter
            filters = self.filter_data(tod,sky_data_flag,500)

            # Next bin into maps
            self.logger(f'{fname}:{self.name}: Creating maps')
            self.maps = self.create_maps(data,tod,filters,sky_data_flag)

            def limfunc(P):
                A,x0,sigx,y0,sigy,phi,B = P
                if (sigx < 0) | (sigy < 0):
                    return True
                if (phi < -np.pi/2.) | (phi >= np.pi/2.):
                    return True
                return False

            self.model_fits = {}
            if self.fit_alt_scan:
                self.logger(f'{fname}:{self.name}: Fitting alternate scans')
                self.model_fits['CW'] = self.fit_source(data, self.maps['CW'] ,limfunc=limfunc)
                self.model_fits['CCW'] = self.fit_source(data, self.maps['CCW'],limfunc=limfunc)

            self.logger(f'{fname}:{self.name}: Fitting global source offset')
            # First get the positions of the sources from the feed average
            self.model_fits['feed_avg'] = self.fit_source(data, self.maps['feed_avg'],limfunc=limfunc)

            self.logger(f'{fname}:{self.name}: Fitting source bands.')
            # Finally, fit the data in the maps
            def limfunc(P):
                A,sigx,sigy,B = P
                if (sigx < 0) | (sigy < 0):
                    return True
                return False
            self.model_fits['maps'] = self.fit_source(data, self.maps['maps'],limfunc=limfunc,fixed_parameters={'x0':True,'y0':True,'phi':True})
        else:
            self.nodata = True
            return


    def fit_source(self,data, maps,limfunc=None,fixed_parameters={}):
        """
        Performs a full fit to the maps obtain the source positions 

        Recommended only for use on the band average data to reduce uncertainties
        """
        fname = data.filename.split('/')[-1]

        self.model.set_fixed(**fixed_parameters)

        avg_map_fits   = {'Values': np.zeros((maps['map'].shape[0],maps['map'].shape[1],self.model.nparams)),
                          'Errors': np.zeros((maps['map'].shape[0],maps['map'].shape[1],self.model.nparams)),
                          'Chi2': np.zeros((maps['map'].shape[0],maps['map'].shape[1],2)),
                          'map_parameters':self.model.get_param_names()}
 
        for ifeed in tqdm(range(maps['map'].shape[0]),desc=f'{self.name}:source_position:{self.source}'):
            for isb in range(maps['map'].shape[1]):
                try:
                    m,c,x,y,P0 = self.prepare_maps(maps['map'][ifeed,isb],maps['cov'][ifeed,isb],maps['xygrid'])
                except AssertionError:
                    continue

                # Perform the least-sqaures fit
                try:
                    result, error,samples, min_chi2, ddof = self.model(P0, (x,y), m, c,return_array=True)
                    avg_map_fits['Values'][ifeed,isb,:] = result
                    avg_map_fits['Errors'][ifeed,isb,:] = error
                    avg_map_fits['Chi2'][ifeed,isb,:]   = min_chi2, ddof
                except ValueError as e:
                    self.logger(f'{fname}:emcee:{e}')
        
        return avg_map_fits
                
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

    def create_maps(self,data,tod,filters,sel):
        """
        Bin maps into instrument frame centred on source
        """

        mjd = data['spectrometer/MJD'][:]

        # We do Jupiter in the Az/El frame but celestial in sky frame
        #if self.source.upper() == 'JUPITER':
        az  = data['spectrometer/pixel_pointing/pixel_az'][:]
        el  = data['spectrometer/pixel_pointing/pixel_el'][:]
        N = az.shape[1]//2 * 2
        daz = np.gradient(az[0,:])*50.
        daz = daz[sel]
        az = az[:,sel]
        el = el[:,sel]
        cw  = daz > 1e-2
        ccw = daz < 1e-2

        mjd=mjd[sel]

        npix = self.Nx*self.Ny

        temp_maps = {'map':np.zeros(npix,dtype=np.float64),
                     'cov':np.zeros(npix,dtype=np.float64)}

        maps = {'maps':{'map':np.zeros((tod.shape[0],tod.shape[1],self.Nx,self.Ny)),
                        'cov':np.zeros((tod.shape[0],tod.shape[1],self.Nx,self.Ny))}}
        maps['feed_avg'] = {'map':np.zeros((tod.shape[0],1,self.Nx,self.Ny)),
                            'cov':np.zeros((tod.shape[0],1,self.Nx,self.Ny))}
        maps['CW'] = {'map':np.zeros((1,1,self.Nx,self.Ny)),
                      'cov':np.zeros((1,1,self.Nx,self.Ny))}
        maps['CCW'] = {'map':np.zeros((1,1,self.Nx,self.Ny)),
                      'cov':np.zeros((1,1,self.Nx,self.Ny))}

        selections = {k:selection for k, selection in zip(maps.keys(),[np.ones(az.shape[-1],dtype=bool),np.ones(az.shape[-1],dtype=bool),cw, ccw])}
        slices = {k:sl for k, sl in zip(maps.keys(),[lambda ifeed,isb:[slice(ifeed,ifeed+1),slice(isb,isb+1),slice(None),slice(None)],
                                                     lambda ifeed,isb:[slice(ifeed,ifeed+1),slice(None),slice(None),slice(None)],
                                                     lambda ifeed,isb:[slice(None),slice(None),slice(None),slice(None)],
                                                     lambda ifeed,isb:[slice(None),slice(None),slice(None),slice(None)]])}

        self.source_positions = {k:a for k,a in zip(['az','el','ra','dec'],Coordinates.sourcePosition(self.source, mjd, self.lon, self.lat))}
        self.source_positions['mean_el'] = np.mean(self.source_positions['el'])
        self.source_positions['mean_az'] = np.mean(self.source_positions['az'])

        for ifeed in tqdm(self.feedlist,desc=f'{self.name}:create_maps:{self.source}'):
            feed_tod = tod[ifeed,...] 

            pixels = self.get_pixel_positions(self.source_positions['az'],self.source_positions['el'],az[ifeed,:],el[ifeed,:])

            mask = np.ones(pixels.size,dtype=int)
            for isb in range(tod.shape[1]):
                for k in temp_maps.keys():
                    temp_maps[k][:] = 0.
                z =  (feed_tod[isb,sel]-filters[ifeed,isb])
                mask[:] = 1
                mask[(pixels == -1) | np.isnan(z) | np.isinf(z)] = 0
                    
                if np.sum(np.isfinite(z)) == 0:
                    continue
                    
                rms = stats.AutoRMS(z)

                weights = {'map':z.astype(np.float64)/rms**2,
                           'cov':np.ones(z.size)/rms**2}
                for k in temp_maps.keys():
                    for mode,map_data in maps.items():
                        if ('CW' in mode) & (ifeed > 1):
                            continue
                        binFuncs.binValues(temp_maps[k], pixels[selections[mode]], weights=weights[k][selections[mode]],mask=mask[selections[mode]])
                        maps[mode][k][slices[mode](ifeed,isb)] = np.reshape(temp_maps[k],(self.Ny,self.Nx))

        xygrid = np.meshgrid((np.arange(self.Nx)+0.5)*self.dx - self.Nx*self.dx/2.,
                             (np.arange(self.Ny)+0.5)*self.dy - self.Ny*self.dy/2.)

        for k,v in maps.items():
            maps[k] = self.average_maps(maps[k])
            maps[k]['xygrid'] = xygrid
        return maps

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

        pX = ((x+Dx/2.)/dx).astype(int)
        pY = ((y+Dy/2.)/dy).astype(int)
        pixels = pX + pY*Nx
        pixels[((pX < 0) | (pX >= Nx)) | ((pY < 0) | (pY >= Ny))] = -1

        return pixels,((x+Dx/2.)/dx),((y+Dy/2.)/dy)
        
    def get_pixel_positions(self,azSource, elSource,az,el):
        x,y =Coordinates.Rotate(azSource, elSource,
                                az,el ,0)

        pixels,pX,pY = self.getpixels(x,y,self.dx,self.dy,self.Nx,self.Ny)
        return pixels

    def filter_data(self,tod,sel,medfilt_size):
        """
        Generate an aggressive median filter to remove large-scale correlated noise.
        """
        
        filters = np.zeros((tod.shape[0],tod.shape[1],int(np.sum(sel))))
        for ifeed in tqdm(self.feedlist,desc=f'{self.name}:filters:{self.source}'):
            feed_tod = tod[ifeed,...] 
            for isb in range(tod.shape[1]):
                z = feed_tod[isb,sel]
                bad = np.where(np.isnan(z))[0]
                if len(bad) == len(z):
                    continue
                if len(bad) > 0:
                    good = np.where(np.isfinite(z))[0]
                        
                nearest = [good[np.argmin(np.abs(good-b))] for b in bad]
                z[bad] = z[nearest]
                filters[ifeed,isb,:] = median_filter(z,medfilt_size)
                    
        return filters
                    

        
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
        if os.path.exists(outfile):
            os.remove(outfile)

        output = h5py.File(outfile,'a')
                  
        if 'Maps' in output:
            del output['Maps']
        map_grp = output.create_group('Maps')
        for mode, maps in self.maps.items():
            if mode in map_grp:
                del map_grp[mode]
            grp = map_grp.create_group(mode)

            dnames = ['Maps','Covariances']
            dsets  = [maps['map'],maps['cov']]
            for (dname, dset) in zip(dnames, dsets):
                if dname in grp:
                    del grp[dname]
                grp.create_dataset(dname,  data=dset)
            grp['Maps'].attrs['Unit'] = 'K'
            grp['Maps'].attrs['cdeltx'] = self.dx
            grp['Maps'].attrs['cdelty'] = self.dy
            grp['Covariances'].attrs['Unit'] = 'K2'
            grp['Covariances'].attrs['cdeltx'] = self.dx
            grp['Covariances'].attrs['cdelty'] = self.dy

        if 'Fits' in output:
            del output['Fits']
        fit_grp = output.create_group('Fits')

        for mode, fits in self.model_fits.items():
            if mode in fit_grp:
                del fit_grp[mode]
            grp = fit_grp.create_group(mode)
            dnames = fits['map_parameters']
            for k in ['Values','Errors']:
                if k in grp:
                    del grp[k]
                grp_vals = grp.create_group(k)
                dsets = [fits[k][...,iparam] for iparam in range(fits[k].shape[-1])]
                for (dname, dset) in zip(dnames, dsets):
                    if dname in output:
                        del output[dname]
                    gauss_dset = grp_vals.create_dataset(dname,  data=dset)
                    gauss_dset.attrs['Unit'] = units[dname]
        

        output.attrs['source'] = self.getSource(data)
        output.close()
