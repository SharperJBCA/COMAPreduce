import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs
from scipy.optimize import fmin, leastsq
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust

from astropy import wcs as wcsModule
import h5py

#from mpi4py import MPI 
#comm = MPI.COMM_WORLD

import os
import healpy as hp
import scipy.fftpack as sfft
from scipy import linalg as la

from tqdm import tqdm

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


class SimulateObservation(DataStructure):
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

class SimulateNoise(DataStructure):
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

class SimulateWhiteNoise(DataStructure):
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

class SimulateSignal(DataStructure):
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
