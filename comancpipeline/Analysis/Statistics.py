import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats
from os import listdir, getcwd
from os.path import isfile, join
from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import pandas as pd
#from mpi4py import MPI
import os
#comm = MPI.COMM_WORLD
from scipy.optimize import minimize, curve_fit

from tqdm import tqdm

class RepointEdges(DataStructure):
    """
    Scan Edge Split - Each time the telescope stops to repoint this is defined as the edge of a scan
    """

    def __init__(self, **kwargs):

        self.max_el_current_fraction = 0.7
        self.min_sample_distance = 10
        self.min_scan_length = 5000 # samples
        self.offset_length = 50
        for item, value in kwargs.items():
            self.__setattr__(item,value)

    def __call__(self, data):
        """
        Expects a level 2 data structure
        """
        return self.getScanPositions(data)

    def getScanPositions(self, d):
        """
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,
        provides indices for the positions of scans in masked array

        Notes:
        - We may need to check for vane position too
        - Iteratively finding the best current fraction may also be needed
        """
        features = d['level1/spectrometer/features'][:]
        uf, counts = np.unique(features,return_counts=True) # select most common feature
        ifeature = np.floor(np.log10(uf[np.argmax(counts)])/np.log10(2))
        selectFeature = self.featureBits(features.astype(float), ifeature)
        index_feature = np.where(selectFeature)[0]

        # make it so that you have a gap, only use data where the telescope is moving

        # Elevation current seems a good proxy for finding repointing times
        elcurrent = np.abs(d['level1/hk/antenna0/driveNode/elDacOutput'][:])
        elutc = d['level1/hk/antenna0/driveNode/utc'][:]
        mjd = d['level1/spectrometer/MJD'][:]

        # these are when the telescope is changing position
        select = np.where((elcurrent > np.max(elcurrent)*self.max_el_current_fraction))[0]

        dselect = select[1:]-select[:-1]
        large_step_indices = np.where((dselect > self.min_sample_distance))[0]

        ends = select[np.append(large_step_indices,len(dselect)-1)]

        # Now map these indices to the original indices
        scan_edges = []
        for (start,end) in zip(ends[:-1],ends[1:]):
            tstart,tend = np.argmin((mjd-elutc[start])**2),np.argmin((mjd-elutc[end])**2)

            # Need to check we are not in a bad feature region
            if selectFeature[tstart] == 0:
                tstart = index_feature[np.argmin((index_feature - tstart)**2)]
            if selectFeature[tend] == 0:
                tend = index_feature[np.argmin((index_feature - tend)**2)]

            if (tend-tstart) > self.min_scan_length:
                Nsteps = int((tend-tstart)//self.offset_length)
                scan_edges += [[tstart,tstart+self.offset_length*Nsteps]]

        return scan_edges



class ScanEdges(DataStructure):
    """
    Splits up observations into "scans" based on parameter inputs
    """

    def __init__(self, scan_edge_type='RepointEdges',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        self.scan_edges = None

        self.scan_edge_type = scan_edge_type

        # Create a scan edge object
        self.scan_edge_object = globals()[self.scan_edge_type](**kwargs)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        allowed_sources = ['fg{}'.format(i) for i in range(10)] + ['GField{:02d}'.format(i) for i in range(20)]
        source  = data['level1/comap'].attrs['source'].decode('utf-8')
        comment = data['level1/comap'].attrs['comment'].decode('utf-8')
        print('SOURCE', source)
        if not source in allowed_sources:
            return data
        if 'Sky nod' in comment:
            return data

        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        self.run(data)
        self.write(data)

        return data

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """

        # Pass data to the scan object to calculate the scan edges
        self.scan_edges = self.scan_edge_object(data)

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """

        if not 'level2' in data:
            return

        lvl2 = data['level2']
        if not 'Statistics' in lvl2:
            statistics = lvl2.create_group('Statistics')
        else:
            statistics = lvl2['Statistics']

        dnames = ['scan_edges']
        dsets = [np.array(self.scan_edges).astype(int)]
        for (dname, dset) in zip(dnames, dsets):
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)


class FnoiseStats(DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, nbins=50, samplerate=50):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        self.nbins = 50
        self.samplerate=50

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod = data['level2/averaged_tod'][...]
        az = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        el = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]
        scan_edges = data['level2/Statistics/scan_edges'][...]


        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape
        nScans = len(scan_edges)

        self.powerspectra = np.zeros((nFeeds, nBands, nChannels, nScans, self.nbins))
        self.freqspectra = np.zeros((nFeeds, nBands, nChannels, nScans, self.nbins))
        self.fnoise_fits = np.zeros((nFeeds, nBands, nChannels, nScans, 2))
        self.wnoise_auto = np.zeros((nFeeds, nBands, nChannels, nScans, 1))
        self.atmos = np.zeros((nFeeds, nBands, nScans, 3))
        self.atmos_errs = np.zeros((nFeeds, nBands, nScans, 3))

        pbar = tqdm(total=((nFeeds-1)*nBands*nChannels*nScans))

        import time
        for ifeed in range(nFeeds):
            if feeds[ifeed] == 20:
                continue

            for iscan,(start,end) in enumerate(scan_edges):
                for iband in range(nBands):

                    tod_filter,atmos,atmos_errs = self.FitAtmosAndGround(np.nanmean(tod[ifeed,iband,:,start:end],axis=0),
                                                                         az[ifeed,start:end],
                                                                         el[ifeed,start:end])

                    self.atmos[ifeed,iband,iscan,:] = atmos
                    self.atmos_errs[ifeed,iband,iscan,:] = atmos_errs

                    for ichan in range(nChannels):
                        if np.nansum(tod[ifeed, iband, ichan,:]) == 0:
                            continue
                        temp = tod[ifeed,iband,ichan,start:end] - tod_filter
                        temp -= np.nanmedian(temp)
                        ps, nu, f_fits, w_auto = self.FitPowerSpectrum(temp)
                        self.powerspectra[ifeed,iband,ichan,iscan,:] = ps
                        self.freqspectra[ifeed,iband,ichan,iscan,:]  = nu
                        self.fnoise_fits[ifeed,iband,ichan,iscan,:]  = f_fits
                        self.wnoise_auto[ifeed,iband,ichan,iscan,:]  = w_auto
                        pbar.update(1)

        #for ifeed in range(nFeeds):
        #    pyplot.errorbar(np.arange(nScans),self.atmos[ifeed,0,:,1],fmt='.',yerr=self.atmos_errs[ifeed,0,:,1])
        #pyplot.show()

        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        allowed_sources = ['fg{}'.format(i) for i in range(10)] + ['GField{:02d}'.format(i) for i in range(20)]
        source = data['level1/comap'].attrs['source'].decode('utf-8')
        comment = data['level1/comap'].attrs['comment'].decode('utf-8')
        print('SOURCE', source)
        if not source in allowed_sources:
            return data
        if 'Sky nod' in comment:
            return data

        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        self.run(data)
        self.write(data)

        return data


    def AutoRMS(self, tod):
        """
        Calculate auto-pair subtracted RMS of tod
        """
        #N2 = tod.size//2*2
        #diff = tod[1:N2:2]-tod[0:N2:2]
        N4 = tod.size//4*4
        ABBA = tod[0:N4:4] - tod[1:N4:4] - tod[2:N4:4] + tod[3:N4:4]
        med = np.nanmedian(ABBA)
        mad = np.sqrt(np.nanmedian(np.abs(ABBA-med)**2))*1.4826/np.sqrt(4)
        return mad

    def PowerSpectrum(self, tod):
        """
        Calculates the bin averaged power spectrum
        """
        nu = np.fft.fftfreq(tod.size, d=1/self.samplerate)
        binEdges = np.logspace(np.log10(nu[1]), np.log10(nu[nu.size//2-1]), self.nbins+1)
        ps     = np.abs(np.fft.fft(tod))**2/tod.size
        counts = np.histogram(nu[1:nu.size//2], binEdges)[0]
        signal = np.histogram(nu[1:nu.size//2], binEdges, weights=ps[1:nu.size//2])[0]
        freqs  = np.histogram(nu[1:nu.size//2], binEdges, weights=nu[1:nu.size//2])[0]

        return freqs/counts, signal/counts, counts

    def Model(self, P, x, rms):
        return rms**2 * (1 + (x/10**P[0])**P[1])

    def Error(self, P, x, y,e, rms):
        error = np.abs(y/e)
        chi = (np.log(y) - np.log(self.Model(P,x,rms)))/error
        return np.sum(chi**2)

    def FitPowerSpectrum(self, tod):
        """
        Calculate the power spectrum of the data, fits a 1/f noise curve, returns parameters
        """
        auto_rms = self.AutoRMS(tod)
        nu, ps, counts = self.PowerSpectrum(tod)

        # Only select non-nan values
        # You may want to increase min counts,
        # as the power spectrum is non-gaussian for small counts
        good = (counts > 50) & ( (nu < 0.03) | (nu > 0.05))

        args = (nu[good], ps[good],auto_rms/np.sqrt(counts[good]), auto_rms)
        bounds =  [[None,None],[-3,0]]
        P0 = [0,-1]
        P1 = minimize(self.Error, P0, args= args, bounds = bounds)


        return ps, nu, P1.x, auto_rms

    def FitAtmosAndGround(self,tod,az,el,niter=100):
        # Fit gradients
        dlength = tod.size

        templates = np.ones((3,tod.size))
        templates[0,:] = az
        if np.abs(np.max(az)-np.min(az)) > 180:
            high = templates[0,:] > 180
            templates[0,high] -= 360
        templates[0,:] -= np.median(templates[0,:])
        templates[1,:] = 1./np.sin(el*np.pi/180.)

        a_all = np.zeros((niter,templates.shape[0]))

        for a_iter in range(niter):
            sel = np.random.uniform(low=0,high=dlength,size=dlength).astype(int)

            cov = np.sum(templates[:,None,sel] * templates[None,:,sel],axis=-1)
            z = np.sum(templates[:,sel]*tod[None,sel],axis=1)
            try:
                a_all[a_iter,:] = np.linalg.solve(cov, z).flatten()
            except:
                a_all[a_iter,:] = np.nan

        fits,errs =  np.nanmedian(a_all,axis=0),stats.MAD(a_all,axis=0)
        tod_filter = np.sum(templates[:,:]*fits[:,None],axis=0)
        return tod_filter, fits, errs


    def RemoveAtmosphere(self, tod, el):
        """
        Remove 1/sin(E) relationship from TOD
        """
        A = 1/np.sin(el*np.pi/180) # Airmass
        pmdl = np.poly1d(np.polyfit(A, tod,1))
        return tod- pmdl(A), pmdl

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """

        if not 'level2' in data:
            return
        lvl2 = data['level2']
        if not 'Statistics' in lvl2:
            statistics = lvl2.create_group('Statistics')
        else:
            statistics = lvl2['Statistics']

        dnames = ['fnoise_fits','wnoise_auto', 'powerspectra','freqspectra', 'atmos','atmos_errors']
        dsets = [self.fnoise_fits,self.wnoise_auto,self.powerspectra,self.freqspectra,self.atmos,self.atmos_errs]
        for (dname, dset) in zip(dnames, dsets):
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)


class SkyDipStats(DataStructure):
    """
    Takes level 1 skydip files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, nbins=50, samplerate=50, min_elevation=0,max_elevation=90):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        self.nbins = 50
        self.samplerate=50

        self.min_elevation = min_elevation
        self.max_elevation = max_elevation

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod = data['level2/averaged_tod'][...]
        az = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        el = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]

        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape

        self.atmos = np.zeros((nFeeds, nBands, nChannels, 2))
        self.atmos_errs = np.zeros((nFeeds, nBands, nChannels, 2))

        pbar = tqdm(total=((nFeeds-1)*nBands*nChannels))

        import time
        for ifeed in range(nFeeds):
            if feeds[ifeed] == 20:
                continue

            elevation_select = (el[ifeed,:] >= self.min_elevation) & (el[ifeed,:] < self.max_elevation)
            for iband in range(nBands):

                for ichan in range(nChannels):
                    atmos,atmos_errs = self.FitAtmos(tod[ifeed,iband,ichan,elevation_select],
                                                     el[ifeed,elevation_select])
                    
                    self.atmos[ifeed,iband,ichan]      = atmos
                    self.atmos_errs[ifeed,iband,ichan] = np.sqrt(np.diag(atmos_errs))
                    
                    
                    pbar.update(1)

        #for ifeed in range(nFeeds):
        #    pyplot.errorbar(np.arange(nScans),self.atmos[ifeed,0,:,1],fmt='.',yerr=self.atmos_errs[ifeed,0,:,1])
        #pyplot.show()

        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        # Ensure that file is a skydip observation
        allowed_sources = ['fg{}'.format(i) for i in range(10)] + ['GField{:02d}'.format(i) for i in range(20)]
        source = data['level1/comap'].attrs['source'].decode('utf-8')
        t_len = int(data['level1/comap'].attrs['nint'])
        comment = data['level1/comap'].attrs['comment'].decode('utf-8')

        print('SOURCE', source)
        if not source in allowed_sources:
            return data
        if not 'Sky nod' in comment:
            return data

        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        self.run(data)
        self.write(data)

        return data

    def model(x,a,b):
        return a*(1-np.exp(b*x))

    def FitAtmos(self,tod,el,niter=100):
        # Fit gradients
        dlength = np.argmax(el[0,:])+1

        cosec_el = 1./np.sin(el*np.pi/180.)

        a_all = np.zeros((niter,2))

        for a_iter in range(niter):
            sel = np.random.uniform(low=0,high=dlength,size=dlength).astype(int)

            try:
                popt, pcov = curve_fit(model, cosec_el[sel], tod[sel])
                a_all[a_iter,:] = popt
            except:
                a_all[a_iter,:] = np.nan

        fits,errs =  np.nanmedian(a_all,axis=0),stats.MAD(a_all,axis=0)
        return fits, errs

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """

        if not 'level2' in data:
            return
        lvl2 = data['level2']
        if not 'SkyDip' in lvl2:
            statistics = lvl2.create_group('SkyDip')
        else:
            statistics = lvl2['SkyDip']

        dnames = ['Opacity', 'Opacity_err', 'atmTemp', 'atmTemp_err']
        dsets = [self.atmos[:,:,:,1], self.atmos_errs[:,:,:,1], self.atmos[:,:,:,0], self.atmos_errs[:,:,:,0]]
        for (dname, dset) in zip(dnames, dsets):
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)
