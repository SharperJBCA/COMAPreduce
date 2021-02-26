import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats
from comancpipeline.Tools.median_filter import medfilt

from os import listdir, getcwd
from os.path import isfile, join
from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import pandas as pd
#from mpi4py import MPI
import os
#comm = MPI.COMM_WORLD
from scipy.optimize import minimize

from tqdm import tqdm

__version__='v1'

def AtmosGroundModel(fits,az,el):
    """
    """
    dlength = az.size

    templates = np.ones((3,az.size))
    templates[0,:] = az
    if np.abs(np.max(az)-np.min(az)) > 180:
        high = templates[0,:] > 180
        templates[0,high] -= 360
    templates[0,:] -= np.median(templates[0,:])
    templates[1,:] = 1./np.sin(el*np.pi/180.)

    tod_filter = np.sum(templates[:,:]*fits[:,None],axis=0)
    return tod_filter



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

    def __init__(self, level2='level2',scan_edge_type='RepointEdges',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'ScanEdges'
        self.scan_edges = None

        self.scan_edge_type = scan_edge_type

        # Create a scan edge object
        self.scan_edge_object = globals()[self.scan_edge_type](**kwargs)
        self.level2 = level2

    def __str__(self):
        return "Scan Edges."

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        allowed_sources = ['fg{}'.format(i) for i in range(10)] +\
                          ['GField{:02d}'.format(i) for i in range(40)] +\
                          ['Field{:02d}'.format(i) for i in range(40)] +\
                          ['Field11b']

        source  = self.getSource(data)
        comment = self.getComment(data)

        if (f'{self.level2}/Statistics' in data) & (not self.overwrite):
            return data

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if self.checkAllowedSources(data, source, allowed_sources):
            return data

        if 'Sky nod' in comment:
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Defining scan edges with {self.scan_edge_type}')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing scan edges to level 2 file ({fname}).')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

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
        fname = data.filename.split('/')[-1]

        if not self.level2 in data:
            self.logger(f'{fname}:{self.name}: No {self.level2} data found?')
            return

        lvl2 = data[self.level2]
        if not 'Statistics' in lvl2:
            self.logger(f'{fname}:{self.name}: Creating Statistics group.')
            statistics = lvl2.create_group('Statistics')
        else:
            self.logger(f'{fname}:{self.name}: Statistics group exists.')
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

    def __init__(self, nbins=50, samplerate=50, medfilt_stepsize=5000,level2='level2',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'FnoiseStats'
        self.nbins = int(nbins)
        self.samplerate = samplerate
        self.medfilt_stepsize = int(medfilt_stepsize)
        self.level2=level2
    def __str__(self):
        return "Calculating noise statistics."

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod = data[f'{self.level2}/averaged_tod'][...]
        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][...]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]
        bands = [b.decode('ascii') for b in data['level1/spectrometer/bands'][:]]

        statistics = self.getGroup(data,data,f'{self.level2}/Statistics')
        scan_edges = self.getGroup(data,statistics,'scan_edges')


        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape
        #if 20 in feeds:
        #    nFeeds -= 1
        nScans = len(scan_edges)

        self.powerspectra = np.zeros((nFeeds, nBands, nScans, self.nbins))
        self.freqspectra = np.zeros((nFeeds, nBands, nScans, self.nbins))
        self.fnoise_fits = np.zeros((nFeeds, nBands, nScans, 3))
        self.wnoise_auto = np.zeros((nFeeds, nBands, nChannels, nScans, 1))
        self.atmos = np.zeros((nFeeds, nBands, nScans, 3))
        self.atmos_errs = np.zeros((nFeeds, nBands, nScans, 3))

        self.filter_tods = [] # Store as a list of arrays, one for each "scan"
        self.filter_coefficients = np.zeros((nFeeds, nBands, nChannels, nScans, 1)) # Stores the per channel gradient of the  median filter
        self.atmos_coefficients = np.zeros((nFeeds, nBands, nChannels, nScans, 1)) # Stores the per channel gradient of the  median filter

        pbar = tqdm(total=(nFeeds*nBands*nChannels*nScans),desc=self.name)

        for iscan,(start,end) in enumerate(scan_edges):
            local_filter_tods = np.zeros((nFeeds,nBands, end-start))
            for ifeed in range(nFeeds):
                if feeds[ifeed] == 20:
                    continue
                for iband in range(nBands):

                    band_average = np.nanmean(tod[ifeed,iband,3:-3,start:end],axis=0)
                    atmos_filter,atmos,atmos_errs = self.FitAtmosAndGround(band_average ,
                                                                         az[ifeed,start:end],
                                                                         el[ifeed,start:end])

                    local_filter_tods[ifeed,iband,:] = self.median_filter(band_average-atmos_filter)[:band_average.size]

                    self.atmos[ifeed,iband,iscan,:] = atmos
                    self.atmos_errs[ifeed,iband,iscan,:] = atmos_errs

                    ps, nu, f_fits, w_auto = self.FitPowerSpectrum(band_average-atmos_filter-local_filter_tods[ifeed,iband,:])
                    self.powerspectra[ifeed,iband,iscan,:] = ps
                    self.freqspectra[ifeed,iband,iscan,:]  = nu
                    self.fnoise_fits[ifeed,iband,iscan,0]  = w_auto
                    self.fnoise_fits[ifeed,iband,iscan,1:] = f_fits

                    #self.logger(f'{fname}:{self.name}: Feed {feeds[ifeed]} Band {bands[iband]} RMS  - {w_auto:.3f}K')
                    #self.logger(f'{fname}:{self.name}: Feed {feeds[ifeed]} Band {bands[iband]} Knee - {f_fits[0]:.3f}')
                    #self.logger(f'{fname}:{self.name}: Feed {feeds[ifeed]} Band {bands[iband]} Spec - {f_fits[1]:.3f}')

                    for ichan in range(nChannels):
                        if np.nansum(tod[ifeed, iband, ichan,start:end]) == 0:
                            continue
                        # Check atmosphere coefficients
                        atmos_coeff,med_coeff,offset = self.coefficient_jointfit(tod[ifeed,iband,ichan,start:end], 
                                                                                 atmos_filter,
                                                                                 local_filter_tods[ifeed,iband,:])
                        w_auto = stats.AutoRMS(tod[ifeed,iband,ichan,start:end])
                        self.wnoise_auto[ifeed,iband,ichan,iscan,:]  = w_auto
                        self.filter_coefficients[ifeed,iband,ichan,iscan,:] = med_coeff
                        self.atmos_coefficients[ifeed,iband,ichan,iscan,:]  = atmos_coeff
                        pbar.update(1)
            self.filter_tods += [local_filter_tods]
        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        allowed_sources = ['fg{}'.format(i) for i in range(10)] +\
                          ['GField{:02d}'.format(i) for i in range(40)] +\
                          ['Field{:02d}'.format(i) for i in range(40)] +\
                          ['Field11b']

        source = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if self.checkAllowedSources(data, source, allowed_sources):
            return data

        if 'Sky nod' in comment:
            return data

        if ('level2/Statistics/fnoise_fits' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Measuring noise stats.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing noise stats to level 2 file ({fname})')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def get_filter_coefficient(self,tod,median_filter):
        """
        Calculate the gradient between tod and filter
        """
        #print('TOD {}, FILTER {}'.format(tod.shape,median_filter.shape))
        return np.sum(tod*median_filter)/np.sum(median_filter**2)

    def coefficient_jointfit(self, tod, atmos, med_filt):
        """
        """
        templates = np.ones((3,tod.size))
        templates[0,:] = atmos
        templates[1,:] = med_filt
        C = templates.dot(templates.T)
        z = templates.dot(tod[:,None])

        a = np.linalg.solve(C,z)
        return a.flatten()


    def median_filter(self,tod):
        """
        Calculate this AFTER removing the atmosphere.
        """
        filter_tod = np.array(medfilt.medfilt(tod.astype(np.float64),np.int32(self.medfilt_stepsize)))

        return filter_tod[:tod.size]

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
        auto_rms = stats.AutoRMS(tod)
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
        fname = data.filename.split('/')[-1]

        if not self.level2 in data:
            return
        lvl2 = data[self.level2]
        if not 'Statistics' in lvl2:
            statistics = lvl2.create_group('Statistics')
        else:
            statistics = lvl2['Statistics']

        dnames = ['fnoise_fits','wnoise_auto', 'powerspectra','freqspectra',
                  'atmos','atmos_errors','filter_coefficients','atmos_coefficients']
        dsets = [self.fnoise_fits,self.wnoise_auto,self.powerspectra,self.freqspectra,
                 self.atmos,self.atmos_errs,self.filter_coefficients,self.atmos_coefficients]
        for (dname, dset) in zip(dnames, dsets):
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)

        # Need to write filter_tods per scan
        for iscan,dset in enumerate(self.filter_tods):
            dname = 'FilterTod_Scan{:02d}'.format(iscan)
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)
            statistics[dname].attrs['medfilt_stepsize'] = self.medfilt_stepsize


class SkyDipStats(DataStructure):
    """
    Takes level 2 files, bins and calibrates them for continuum analysis.
    Does not require scan_edges to run
    """

    def __init__(self, nbins=50, samplerate=50, medfilt_stepsize=5000, poly_iter=100, dipLo=42, dipHi=58):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        poly_iter - how many times to bootstrap 90% of values from
        """
        self.nbins = int(nbins)
        self.samplerate = samplerate
        self.medfilt_stepsize = int(medfilt_stepsize)
        self.poly_iter = int(poly_iter)
        self.dipLo = int(dipLo)
        self.dipHi = int(dipHi)

    def __str__(self):
        return "Calculating noise statistics (skydip class)."

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod = data['level2/averaged_tod'][...]
        az = data['level1/spectrometer/pixel_pointing/pixel_az'][...]
        el = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]
        feat = data['level1/spectrometer/features'][...]


        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape
        self.opacity = np.zeros((nFeeds, nBands, nChannels))
        self.opacity_err = np.zeros((nFeeds, nBands, nChannels))
        self.Tzen = np.zeros((nFeeds, nBands, nChannels))
        self.Tzen_err = np.zeros((nFeeds, nBands, nChannels))

        pbar = tqdm(total=((nFeeds-1)*nBands*nChannels))

        skydip_select = np.all([tod_skydip>self.dipLo,
                                tod_skydip<self.dipHi,
                                feat==256],
                               axis=0)
        import time

        for ifeed in range(nFeeds):
            if feeds[ifeed] == 20:
                continue
            for iband in range(nBands):

                for ichan in range(nChannels):
                    x = 1/(np.cos(el[ifeed,skydip_select[ifeed]]*(np.pi/180)))
                    y = tod_skydip[ifeed,iband,ichan,skydip_select[ifeed]]

                    total = np.shape(x)[0]
                    boot_no = int(np.rint(total*0.9))
                    coeffs = np.zeros((self.poly_iter,2))
                    coeffs[:] = np.nan
                    if np.all(np.isnan(y))==False:
                        for n in range(self.poly_iter):
                            boot_sel = np.random.randint(0,high=total,size=boot_no)
                            try:
                                coeffs[n] = np.polyfit(x[boot_sel],y[boot_sel],1)
                            except:
                                pass

                        avg = np.nanmean(coeffs,axis=1)
                        std = np.nanstd(coeffs,axis=1)
                    else:
                        avg = np.asarray((np.nan,np.nan))
                        std = np.asarray((np.nan,np.nan))

                    #assume Tatm=300K
                    self.opacity[ifeed,iband,ichan] = avg[1]/300#K
                    self.opacity_err[ifeed,iband,ichan] = std[1]/300#K
                    self.Tzen[ifeed,iband,ichan] = avg[0]
                    self.Tzen_err[ifeed,iband,ichan] = std[0]

                    pbar.update(1)

        #for ifeed in range(nFeeds):
        #    pyplot.errorbar(np.arange(nScans),self.atmos[ifeed,0,:,1],fmt='.',yerr=self.atmos_errs[ifeed,0,:,1])
        #pyplot.show()

        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        allowed_sources = ['fg{}'.format(i) for i in range(10)] +\
                          ['GField{:02d}'.format(i) for i in range(40)] +\
                          ['Field{:02d}'.format(i) for i in range(40)] +\
                          ['Field11b']

        source = self.getSource(data)
        comment = self.getComment(data)
        if self.checkAllowedSources(data, source, allowed_sources):
            return data

        if not 'Sky nod' in comment:
            return data

        self.run(data)

        # Want to ensure the data file is read/write
        self.setReadWrite(data)

        self.write(data)

        return data

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """

        if not 'level2' in data:
            return
        lvl2 = data['level2']
        if not 'SkyDipStats' in lvl2:
            SkyDipStats = lvl2.create_group('SkyDipStats')
        else:
            SkyDipStats = lvl2['SkyDipStats']

        dnames = ['opacity', 'opacity_err', 'Tzenith', 'Tzenith_err']
        dsets = [self.opacity, self.opacity_err, self.Tzen_err, self.Tzen_err]
        for (dname, dset) in zip(dnames, dsets):
            if dname in SkyDipStats:
                del SkyDipStats[dname]
            SkyDipStats.create_dataset(dname,  data=dset)
