import numpy as np
import comancpipeline
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs, ParserClass
from comancpipeline.Simulations import SkyModel
from scipy.optimize import fmin, leastsq
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot
import shutil

#from comancpipeline.Tools import WCSa 
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust

from astropy import wcs 
import h5py

#from mpi4py import MPI 
#comm = MPI.COMM_WORLD

import os
import healpy as hp
import scipy.fftpack as sfft
from scipy import linalg as la
from astropy.io import fits

from tqdm import tqdm

__vane_version__ = 'v3'
__level2_version__ = 'v1'

def ang2pixWCS(w, phi, theta, image_shape):
    """
    Ang2Pix given a known wcs object

    args:
    wcs : wcs object
    ra : arraylike, degrees
    dec : arraylike, degrees

    returns:
    pixels : arraylike, int
    """

    # Generate pixel coordinates
    pixcrd = np.floor(np.array(w.wcs_world2pix(phi, theta, 0))).astype('int64')

    bd = ((pixcrd[0,:] < 0) | (pixcrd[1,:] < 0)) | ((pixcrd[0,:] >= image_shape[1]) | (pixcrd[1,:] >= image_shape[0])) 

    pix = pixcrd[0,:] + pixcrd[1,:]*int(image_shape[1])
    pix = pix.astype('int')
    pix[bd] = -1

    npix = int(image_shape[0]*image_shape[1])

    return pix

class CreateSimulateLevel2Cont(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1,
                 average_width=512,calvanedir='AncillaryData/CalVanes',
                 cal_mode = 'Vane', cal_prefix='',level2='level2',
                 samplerate=50,
                 data_dirs=None,
                 set_permissions=True,
                 permissions_group='comap',
                 calvane_prefix='CalVane',
                 simulate_signal=True,
                 signal_parameter_file='',
                 signal_models=[],
                 simulate_white_noise=False,
                 simulate_atmosphere=False,
                 signal_map_file='',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)

        self.name = 'CreateLevel2Cont'
        self.feeds_select = feeds

        self.output_dir = output_dir
        if isinstance(data_dirs,type(None)):
            self.data_dirs = [self.output_dir]
        else:
            if isinstance(data_dirs,list):
                self.data_dirs = data_dirs
            else:
                self.data_dirs = [data_dirs]

        self.nworkers = int(nworkers)
        self.average_width = int(average_width)

        self.calvanedir = calvanedir
        self.calvane_prefix = calvane_prefix

        self.cal_mode = cal_mode
        self.cal_prefix=cal_prefix

        self.level2=level2
        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

        # Setup the sky signal information
        self.simulate_signal = simulate_signal
        self.simulate_white_noise = simulate_white_noise
        self.simulate_atmosphere  = simulate_atmosphere
        self.samplerate = samplerate # Hz
        if simulate_signal:
            self.signal_parameters = ParserClass.Parser(signal_parameter_file)
            model_info = {k:self.signal_parameters[k] for k in signal_models}
            self.skymodel = SkyModel.SkyModel(model_info)

    def __str__(self):
        return "Creating simulated level2 file with channel binning of {}".format(self.average_width)

    def run(self,data):
        """
        Sets up feeds that are needed to be called,
        grabs the pointer to the time ordered data,
        and calls the averaging routine in SourceFitting.FitSource.average(*args)

        """
        # Setup feed indexing
        # self.feeds : feed horn ID (in array indexing, only chosen feeds)
        # self.feedlist : all feed IDs in data file (in lvl1 indexing)
        # self.feeddict : map between feed ID and feed array index in lvl1
        self.feeds, self.feed_index, self.feed_dict = self.getFeeds(data,self.feeds_select)

        # Opening file here to write out data bit by bit
        self.i_nFeeds, self.i_nBands, self.i_nChannels,self.i_nSamples = data['level1/spectrometer/tod'].shape
        avg_tod_shape = (self.i_nFeeds, self.i_nBands, self.i_nChannels//self.average_width, self.i_nSamples)
        self.i_nChannels = avg_tod_shape[2]

        frequency = data['level1/spectrometer/frequency'][...]
        self.avg_frequency = np.nanmean(np.reshape(frequency,(self.i_nBands, self.i_nChannels, self.average_width)),axis=2)

        self.avg_tod = np.zeros(avg_tod_shape,dtype=data['level1/spectrometer/tod'].dtype)

        # Average the data and apply the gains
        self.simulate_obs(data, self.avg_tod)

    def sky_signal(self,data,avg_tod):
        """
        1) Read in a sky map
        2) Sample at observed pixels
        3) Return time ordered data
        """

        feeds = np.arange(self.i_nFeeds,dtype=int)

        ra     = data['level1/spectrometer/pixel_pointing/pixel_ra'][...]
        dec    = data['level1/spectrometer/pixel_pointing/pixel_dec'][...]

        for ifeed in tqdm(feeds.flatten()):
            gl, gb = Coordinates.e2g(ra[ifeed], dec[ifeed]) 
            for iband in range(self.i_nBands):
                for ichan in range(self.i_nChannels):
                    avg_tod[ifeed,iband,ichan] += self.skymodel(gl,gb,self.avg_frequency[iband,ichan])

    def atmosphere(self, data, avg_tod, tau=0.01, Tatm=280):
        """
        """
        feeds = np.arange(self.i_nFeeds,dtype=int)
        tauTb = np.nanmedian(data['level2/Statistics/atmos'][0,0,:,1])
        
        el = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        for ifeed in tqdm(feeds.flatten()):
            A      = tauTb/np.sin(np.abs(el[ifeed])*np.pi/180.)
            avg_tod[ifeed] += A #Tatm * ( 1 - np.exp(-tau*A[None,None,:]))

    def white_noise(self, data, avg_tod,Trec=20):
        """
        """
        fnoise_fits = data['level2/Statistics/fnoise_fits'][...]
        fnoise_fits = np.nanmedian(fnoise_fits,axis=(2,3))
        med_fnoise  = np.nanmedian(fnoise_fits,axis=(0,1))
        for i in range(len(med_fnoise)):
            good = np.isfinite(fnoise_fits[...,i])
            fnoise_fits[~good,i] = med_fnoise[i]

        wnoise_rms = data['level2/Statistics/wnoise_auto'][...]
        med_wnoise = np.nanmedian(wnoise_rms)
        bad = (wnoise_rms < 1e-2) | (wnoise_rms > 1.5e-1)
        wnoise_rms[bad] = med_wnoise
        wnoise_rms = np.nanmedian(wnoise_rms,axis=(2,3))
        wnoise_rms[(wnoise_rms == 0)] = np.nanmean(wnoise_rms[wnoise_rms != 0])
        nu = np.fft.fftfreq(avg_tod.shape[-1],d=1./self.samplerate)

        feeds = np.arange(self.i_nFeeds,dtype=int)
        for ifeed in tqdm(feeds.flatten()):
            P = np.sqrt((np.abs(nu[None,:])/10**fnoise_fits[ifeed,...,1:2])**fnoise_fits[ifeed,...,2:3])
            P[:,0]=0.
            noise_model = wnoise_rms[ifeed]*P
            test_noise = np.random.normal(scale=1,size=(avg_tod[ifeed].shape[0],
                                                        avg_tod[ifeed].shape[-1])) # Nbands x Ntod
            test_noise = np.real(np.fft.ifft(np.fft.fft(test_noise,axis=-1)*noise_model[:,:],axis=-1))

            scale = wnoise_rms[ifeed,...,None] * np.ones(avg_tod[ifeed].shape) # Nbands x Nchan X Ntod
            avg_tod[ifeed] += test_noise[:,None,:] + \
                              np.random.normal(scale=scale)


    def simulate_obs(self, data, avg_tod):
        """
        Simulate observations wrapper
        """

        if self.simulate_signal:
            self.sky_signal(data,avg_tod)

        if self.simulate_atmosphere:
            self.atmosphere(data,avg_tod)

        if self.simulate_white_noise:
            self.white_noise(data,avg_tod)

        

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        self.comment = self.getComment(data)
        prefix = os.path.basename(data.filename).split('.hd5')[0]
        self.outfilename = '{}/{}_Level2Sim.hd5'.format(self.output_dir,prefix)

        if os.path.exists(self.outfilename) & (not self.overwrite):
            data.close()
            self.outfile = h5py.File(self.outfilename,'a')
            return self.outfile

        self.logger(f'{fname}:{self.name}: Starting level 2 simulation creation.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing level 2 file: {self.outfilename}')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')
        # Now change to the level2 file for all future analyses.
        if data:
            data.close() # just double check we close the level 1 file.
        return self.outfile

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """
        if not os.path.exists(os.path.dirname(self.outfilename)):
            os.makedirs(os.path.dirname(self.outfilename))

        if os.path.exists(self.outfilename):
            self.outfile = h5py.File(self.outfilename,'a')
        else:
            self.outfile = h5py.File(self.outfilename,'w')

        # Set permissions and group
        if self.set_permissions:
            os.chmod(self.outfilename,0o664)
            shutil.chown(self.outfilename, group=self.permissions_group)

        if self.level2 in self.outfile:
            del self.outfile[self.level2]
        lvl2 = self.outfile.create_group(self.level2)

        tod_dset = lvl2.create_dataset('averaged_tod',data=self.avg_tod, dtype=self.avg_tod.dtype)
        tod_dset.attrs['Unit'] = 'K'
        tod_dset.attrs['Calibration'] = '{self.cal_mode}:{self.cal_prefix}'

        freq_dset = lvl2.create_dataset('frequency',data=self.avg_frequency, dtype=self.avg_frequency.dtype)

        # Link the Level1 data
        data_filename = data['level1'].file.filename
        fname = data['level1'].file.filename.split('/')[-1]
        vane_file = data['level2/Vane'].file.filename

        # Copy over the statistics
        if 'Statistics' in lvl2:
            del lvl2['Statistics']
        grp = lvl2.create_group('Statistics')
        for k,v in data['level2/Statistics'].items():
            if isinstance(v,h5py.Group):
                grp2 = grp.create_group(k)
                for k1,v1 in v.items():
                    grp2.create_dataset(k1,data=v1,dtype=v1.dtype)
            else:
                grp.create_dataset(k,data=v,dtype=v.dtype)


        data.close()
        if 'level1' in self.outfile:
            del self.outfile['level1']
        self.outfile['level1'] = h5py.ExternalLink(data_filename,'/')
        lvl2.attrs['version'] = __level2_version__

        # Add version info
        lvl2.attrs['pipeline-version'] = comancpipeline.__version__

        # Link the Level1 data
        if 'Vane' in lvl2:
            del lvl2['Vane']
        lvl2['Vane'] = h5py.ExternalLink('{}'.format(vane_file),'/')



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

    def amplitude(self, frequency, fwhm):
        a = 4.695
        b = 0.085
        c = -0.178
        logS = a + b*np.log10(frequency*1e3) + c*np.log10(frequency*1e3)**2
        S = 10**logS

        c = 3e8
        kb = 1.38e-23
        T2Jy = 2 * kb * 1e9**2 / c**2 * frequency**2

        return  S / T2Jy / 1.13/ (fwhm*np.pi/180.)**2 * 1e-26
    
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


class SimulateObservation(BaseClasses.DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,outputdir='Simulations/', source='CygA' ):
        self.source = sources[source]
        self.outputdir = outputdir

        self.fwhm = 4.5/60. 
        
        self.tau = 0.1
        self.Tatm= 300

        self.sampleRate = 50.
        self.beta = 1e9

        self.noisecov = np.loadtxt('/scratch/nas_comap1/tjrennie/Stuart/OBS:8502CORMATRIX:BAND0',delimiter=',')
        print(self.noisecov.shape)

    def corrnoise(self,N):
        L,lu1,lu2 = la.lu(self.noisecov)
        z = np.random.normal(size=(17,N))

        def fnoise(tod):
            ftod = sfft.fft(tod,axis=1)
            f = np.fft.fftfreq(tod.shape[1],d=1./self.sampleRate)
            f[0] = f[1]
            P = np.abs(f)**-1.5
            P /= np.sum(P)
            ftod = ftod*np.sqrt(P)[np.newaxis,:]
            return np.real(sfft.ifft(ftod,axis=1))

        z = fnoise(z)
        return lu1.dot(z)

    def __str__(self):
        return 'Simulating source at {:.2f} {:.2f}'.format(self.x0, self.y0)

    def run(self,data):

        self.simulate(data)
        #self.write(data)

    def Gauss2D(self,Amplitude, xcen, ycen, sigma,x , y):
        
        Z = (x-xcen)**2/sigma**2 + (y-ycen)**2/sigma**2
        return Amplitude * np.exp(-0.5*Z)

    def Atmos(self, el):
        return self.Tatm*(1-np.exp(-self.tau/np.sin(el*np.pi/180.)))

    def ReceiverNoise(self,tod):
        sigma = tod/np.sqrt(self.beta/self.sampleRate)
        return np.random.normal(scale=sigma)
        

    def simulate(self,data):
        todshape = data['spectrometer/band_average'].shape
        self.feeds = data['spectrometer/feeds'][:]
        self.mjd = data['spectrometer/MJD'][:]
        self.az  = data['spectrometer/pixel_pointing/pixel_az'][...]
        self.el  = data['spectrometer/pixel_pointing/pixel_el'][...]
        self.ra  = data['spectrometer/pixel_pointing/pixel_ra'][...]
        self.dec = data['spectrometer/pixel_pointing/pixel_dec'][...]

        self.tod = np.zeros(todshape)

        self.source.ra,self.source.dec = Coordinates.precess2year(np.array([self.source.x]),
                                                                  np.array([self.source.y]),
                                                                  np.array([np.mean(self.mjd)]))

        nHorns, nSBs, nSamples = todshape
        rot = hp.rotator.Rotator(rot=[self.source.ra, self.source.dec])
        for j in range(nSBs):
            for i in tqdm(range(nHorns)):
                decRot, raRot = rot((90-self.dec[i,:])*np.pi/180., self.ra[i,:]*np.pi/180.)
                self.tod[i,j,:] += self.Gauss2D(self.source.amplitude(30, self.fwhm),
                                           0,0,self.fwhm/2.355,
                                           raRot*180./np.pi, (np.pi/2.-decRot)*180./np.pi)

                self.tod[i,j,:] += self.Atmos(self.el[i,:])
                self.tod[i,j,:] += self.ReceiverNoise(self.tod[i,j,:])
            ft = self.corrnoise(self.tod.shape[-1])*5
            self.tod[:17,j,:] += ft

    def write(self,data):

        filename = data.filename.split('/')[-1]
        if os.path.exists('{}/{}'.format(self.outputdir,filename)):
            os.remove('{}/{}'.format(self.outputdir,filename))
        out = h5py.File('{}/{}'.format(self.outputdir,filename))

        out.create_dataset('spectrometer/band_average',data=self.tod)
        out.create_dataset('spectrometer/MJD',data=self.mjd)
        out.create_dataset('spectrometer/feeds',data=self.feeds)
        out.create_dataset('spectrometer/features',data=data['spectrometer/features'][...])
        out.create_dataset('spectrometer/pixel_pointing/pixel_az',data=self.az)
        out.create_dataset('spectrometer/pixel_pointing/pixel_el',data=self.el)
        out.create_dataset('spectrometer/pixel_pointing/pixel_ra',data=self.ra)
        out.create_dataset('spectrometer/pixel_pointing/pixel_dec',data=self.dec)
        out.create_group('comap')
        for attr,value in data['comap'].attrs.items():
            out['comap'].attrs[attr] = value
        out.close()

##
from astropy.io import fits
class SimulateDiffuse(SimulateObservation):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,outputdir='Simulations/', source='CygA' ):
        self.source = sources[source]
        self.outputdir = outputdir

        self.fwhm = 4.5/60. 
        
        self.tau = 0.1
        self.Tatm= 300

        self.sampleRate = 50.
        self.beta = 1e9

        self.noisecov = np.loadtxt('/scratch/nas_comap1/tjrennie/Stuart/OBS:8502CORMATRIX:BAND0',delimiter=',')
        self.Trec = 40 * np.sqrt(2)

        # Get the VGPS data
        MJysr2K = 40e-6
        self.skymap = hp.read_map('/local/scratch/sharper/TemplateFittingNew/MAPS/IRIS_combined_SFD_really_nohole_nosource_4_2048.fits')*MJysr2K

    def VGPS(self,ra, dec):
        rot = hp.rotator.Rotator(coord=['C','G'])
        gb, gl = rot((90-dec)*np.pi/180., ra*np.pi/180.)
        gb, gl = (np.pi/2. - gb)*180./np.pi, gl*180./np.pi

        xpix, ypix = np.meshgrid(np.arange(self.vgps.shape[1]),np.arange(self.vgps.shape[0]))

        crval = self.vgps_wcs.crpix[0]*self.vgps_wcs.cdelt[0],self.vgps_wcs.crpix[1]*self.vgps_wcs.cdelt[1] 
        print(crval)


        glpix, gbpix = self.vgps_wcs.all_pix2world(xpix.flatten(),ypix.flatten(),0)
        print(self.vgps_wcs.naxis,self.vgps_wcs.cdelt,self.vgps_wcs.crval,self.vgps_wcs.crpix,self.vgps_wcs.ctype)
        print( self.vgps_wcs.all_pix2world(7113,513,0))
        #pyplot.plot(gl,gb,'.')
        #pyplot.plot(glpix,gbpix,',')
        #pyplot.show()     

    def Diffuse(self,ra, dec):
        rot = hp.rotator.Rotator(coord=['C','G'])
        gb, gl = rot((90-dec)*np.pi/180., ra*np.pi/180.)
        #gb, gl = (np.pi/2. - gb)*180./np.pi, gl*180./np.pi
       # pix = hp.ang2pix(2048,gb,gl)
        
        return hp.get_interp_val(self.skymap, gb,gl) #self.skymap[pix]

    def simulate(self,data):
        todshape = data['spectrometer/band_average'].shape
        self.feeds = data['spectrometer/feeds'][:]
        self.mjd = data['spectrometer/MJD'][:]
        self.az  = data['spectrometer/pixel_pointing/pixel_az'][...]
        self.el  = data['spectrometer/pixel_pointing/pixel_el'][...]
        self.ra  = data['spectrometer/pixel_pointing/pixel_ra'][...]
        self.dec = data['spectrometer/pixel_pointing/pixel_dec'][...]

        self.tod = np.zeros(todshape)

        self.source.ra,self.source.dec = Coordinates.precess2year(np.array([self.source.x]),
                                                                  np.array([self.source.y]),
                                                                  np.array([np.mean(self.mjd)]))

        nHorns, nSBs, nSamples = todshape
        rot = hp.rotator.Rotator(rot=[self.source.ra, self.source.dec])
        for j in range(nSBs):
            for i in tqdm(range(nHorns)):
                decRot, raRot = rot((90-self.dec[i,:])*np.pi/180., self.ra[i,:]*np.pi/180.)
                #self.tod[i,j,:] += self.Gauss2D(self.source.amplitude(30, self.fwhm),
                #                           0,0,self.fwhm/2.355,
                #                           raRot*180./np.pi, (np.pi/2.-decRot)*180./np.pi)

                self.tod[i,j,:] += self.Diffuse(self.ra[i,:],self.dec[i,:])
                self.tod[i,j,:] += self.Trec
                #self.tod[i,j,:] += self.Atmos(self.el[i,:])
                self.tod[i,j,:] += self.ReceiverNoise(self.tod[i,j,:])
            #ft = self.corrnoise(self.tod.shape[-1])/10
            #self.tod[:17,j,:] += ft
            #pyplot.plot(self.tod[0,0,:]-np.nanmedian(self.tod[0,0,:]))
            #pyplot.plot(self.Diffuse(self.ra[0,:],self.dec[0,:]))
            #pyplot.show()

class SimulateNoise(BaseClasses.DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,alpha = 1, tsys=40, fk=1.):
        self.tsys = tsys
        self.fk = fk
        self.alpha = alpha

    def __str__(self):
        return 'Simulating noise with Tsys: {:.1f}'.format(self.tsys) + r' f_knee: '+ '{:.2f}'.format(self.fk)

    def realisation(self, N):
        if not hasattr(self, 'f'):
            self.Nbig = 2**(int(np.log(N)/np.log(2)+1))
            self.f = sfft.fftfreq(self.Nbig, d=self.dT)
            self.f[0] = self.f[1]
        rms = self.tsys/np.sqrt(self.dT * self.dnu) * np.sqrt(2)

        wnoise = np.random.normal(size=self.Nbig, scale=rms)
        power  = (1. + (self.fk/np.abs(self.f))**self.alpha)
        noise  = sfft.ifft(sfft.fft(wnoise)*np.sqrt(power))

        return noise[:N] + self.tsys

    def run(self,data):

        self.simulate(data)

    def simulate(self,data):
        tod = data.getdset('spectrometer/tod')
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9
        
        nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] = self.realisation(nSamples)

class SimulateWhiteNoise(BaseClasses.DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self, tsys=40, rfactor=1):
        '''
        rfactor - reduce noise by a given factor
        '''
        self.tsys = tsys
        self.rfactor = rfactor
        

    def __str__(self):
        return 'Simulating white noise with Tsys: {:.1f}'.format(self.tsys)

    def realisation(self, N):
        rms = self.tsys/np.sqrt(self.dT * self.dnu) * np.sqrt(2) * self.rfactor
        wnoise = np.random.normal(size=N, scale=rms)
        return wnoise + self.tsys

    def run(self,data):

        self.simulate(data)

    def simulate(self,data):
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')

        
        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9

        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            print('Creating new dataset')
            tod = np.zeros((nHorns, nSBs, nChans, nSamples))
        else:
            print('Reading existing dataset')
            tod = data.getdset('spectrometer/tod')

        for i in range(nHorns):
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] += self.realisation(nSamples)

        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            data.updatedset('spectrometer/tod', tod)
            data.setAttr('comap', 'sim_tod', True)

class SimulateSignal(BaseClasses.DataStructure):
    """
    Base source fitting class.

    Contains functions for rotating coordinate system to frame
    of the source being fitted. Useful for aperture photometry and
    Beam fitting functions.
    """

    def __init__(self,cocubefilename=None, ra0=0, dec0=0):
        self.cocubefilename = cocubefilename
        self.ra0 = ra0
        self.dec0=dec0

    def __str__(self):
        return 'Simulating signal from {}'.format(self.cocubefilename)

    def run(self,data):
        self.simulate(data)

    def readCOCube(self):
        '''
        '''
        self.cubedataFile = h5py.File(self.cocubefilename,'r')
        self.wcs = wcsModule.WCS(naxis=2) 
        self.wcs.wcs.crval = [self.ra0, self.dec0]
        self.wcs.wcs.cdelt = self.cubedataFile['cube'].attrs['cdelt']
        self.wcs.wcs.crpix = self.cubedataFile['cube'].attrs['crpix']
        self.wcs.wcs.ctype = [self.cubedataFile['cube'].attrs['ctype1'],
                              self.cubedataFile['cube'].attrs['ctype2'] ]
        self.nxpix = self.cubedataFile['cube'].attrs['nxpix']
        self.nypix = self.cubedataFile['cube'].attrs['nypix']

        self.cocube = self.cubedataFile['cube'][...]

        self.coFreq = self.cubedataFile['frequencies'][:]
        return self.cubedataFile, self.wcs

    def getPixelCoords(self,ra, dec):
        '''
        '''
        ypixels, xpixels = self.wcs.wcs_world2pix(ra,dec,0)
        pflat  = (xpixels.astype(int) + self.nxpix*ypixels.astype(int)).astype(int)
        gd = (xpixels >= 0) & (xpixels < self.nxpix) & (ypixels >= 0) & (ypixels < self.nypix)
        pflat[~gd] = -1
        return pflat, np.where(~(gd))[0]

    def cube2TOD(self, pixels, channel, masked):
        '''
        '''
        tod = self.cocube[pixels, channel]
        tod[masked] = 0
        return tod

    def simulate(self,data):
        '''
        '''
        mjd = data.getdset('spectrometer/MJD')
        nu  = data.getdset('spectrometer/frequency')
        ra  = data.getdset('spectrometer/pixel_pointing/pixel_ra')
        dec = data.getdset('spectrometer/pixel_pointing/pixel_dec')

        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        nSBs, nChans = nu.shape
        nHorns, nSamples = ra.shape
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            print('Creating new dataset')
            tod = np.zeros((nHorns, nSBs, nChans, nSamples))
        else:
            print('Reading existing dataset')
            tod = data.getdset('spectrometer/tod')

        self.dT  = np.abs(mjd[-1] - mjd[0])*86400./mjd.size
        self.dnu = np.abs(nu[0,1] - nu[0,0])*1e9

        cocube, wcs = self.readCOCube()

        # map indices of COCube frequencies to those in real data.
        self.nuMap = [[np.argmin((nui-self.coFreq)**2) for nui in sb] for sb in nu]
        
        #nHorns, nSBs, nChans, nSamples = tod.shape
        for i in range(nHorns):
            pflat, masked = self.getPixelCoords(ra[i,:], dec[i,:])
            for j in range(nSBs):
                for k in range(nChans):
                    tod[i,j,k,:] += self.cube2TOD(pflat, self.nuMap[j][k], masked)
                    
        if isinstance(data.getAttr('comap','sim_tod'), type(None)):
            data.updatedset('spectrometer/tod', tod)
            data.setAttr('comap', 'sim_tod', True)

        self.cubedataFile.close()
