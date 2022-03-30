import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats, FileTools
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
from scipy import linalg
from tqdm import tqdm

from scipy.signal import find_peaks

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



class RepointEdges(BaseClasses.DataStructure):
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

    def __call__(self, data, source=''):
        """
        Expects a level 2 data structure
        """
        if any([source in f for f in ['TauA','CasA','CygA','jupiter','Jupiter']]):
            return self.getBetweenVane(data)
        else:
            return self.getScanPositions(data)


    def getBetweenVane(self,d):
        """
        Defines data as one large scan defined as the start and end of the vane calibration
        """

        return [[int(d['level2/Vane/VaneEdges'][0,1]),
                 int(d['level2/Vane/VaneEdges'][1,0])]]

    def getScanPositions(self, d):
        """
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,
        provides indices for the positions of scans in masked array

        Notes:
        - We may need to check for vane position too
        - Iteratively finding the best current fraction may also be needed
        """
        features = self.getFeatures(d) 
        uf, counts = np.unique(features,return_counts=True) # select most common feature
        ifeature = np.floor(np.log10(uf[np.argmax(counts)])/np.log10(2))
        selectFeature = self.featureBits(features.astype(float), ifeature)
        index_feature = np.where(selectFeature)[0]
        from matplotlib import pyplot
        if ifeature == 9.0:
            #azcurrent = np.abs(d['level1/hk/antenna0/driveNode/azDacOutput'][:])
            elutc = d['level1/spectrometer/MJD'][:] #d['level1/hk/antenna0/driveNode/utc'][:]
            mjd = d['level1/spectrometer/MJD'][:]
            # these are when the telescope is changing position
            #cutpoint = np.max([np.max(azcurrent)*self.max_el_current_fraction,3000])
            #select = np.where((azcurrent > cutpoint))[0]
            ends = np.array([index_feature[0],index_feature[-1]])
        elif ifeature == 5.0:
            # Elevation current seems a good proxy for finding repointing times
            elcurrent = np.abs(d['level1/hk/antenna0/driveNode/elDacOutput'][:])
            elutc = d['level1/hk/antenna0/driveNode/utc'][:]
            mjd = d['level1/spectrometer/MJD'][:]
            cutoff = np.max(elcurrent)*self.max_el_current_fraction
            
            # these are when the telescope is changing position
            select = np.where((elcurrent > cutoff))[0]

            dselect = select[1:]-select[:-1]
            large_step_indices = np.where((dselect > self.min_sample_distance))[0]

            ends = select[np.append(large_step_indices,len(dselect)-1)]

        else:
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



class ScanEdges(BaseClasses.DataStructure):
    """
    Splits up observations into "scans" based on parameter inputs
    """

    def __init__(self, 
                 allowed_sources = ['fg','GField','Field','TauA','CasA','Jupiter','jupiter','Cyga'],
                 level2='level2',
                 scan_edge_type='RepointEdges',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'ScanEdges'
        self.scan_edges = None

        self.scan_edge_type = scan_edge_type
        self.allowed_sources = allowed_sources
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


        self.source  = self.getSource(data)
        comment = self.getComment(data)

        if (f'{self.level2}/Statistics' in data) & (not self.overwrite):
            return data

        self.logger(f'{fname}:{self.name}: {self.source} - {comment}')

        if self.checkAllowedSources(data, self.source, self.allowed_sources):
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
        self.scan_edges = self.scan_edge_object(data,source=self.source)

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


class FnoiseStats(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, allowed_sources = ['fg','GField','Field','TauA','CasA','Jupiter','jupiter','Cyga'],
                 nbins=50, 
                 samplerate=50, 
                 medfilt_stepsize=5000,
                 database = None,
                 level2='level2',**kwargs):
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
        self.allowed_sources = allowed_sources
        self.database = database
    def __str__(self):
        return "Calculating noise statistics."

    def getSourceMask(self,ra,dec,mjd):
        
        if not self.isCalibrator:
            return np.ones(mjd.size,dtype=bool)
        if self.source.lower() == 'jupiter':
            az,el,ra0,dec0=Coordinates.sourcePosition(self.source,mjd, Coordinates.comap_longitude, Coordinates.comap_latitude)
        else:
            ra0, dec0 = Coordinates.CalibratorList[self.source]

        distance = Coordinates.AngularSeperation(ra0,dec0,ra,dec)

        return (distance > 10./60.)

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
        mjd = data['level1/spectrometer/MJD'][...]
        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][...]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        ra  = data['level1/spectrometer/pixel_pointing/pixel_ra'][...]
        dec = data['level1/spectrometer/pixel_pointing/pixel_dec'][...]
        feeds = data['level1/spectrometer/feeds'][:]
        bands = [b.decode('ascii') for b in data['level1/spectrometer/bands'][:]]

        statistics = self.getGroup(data,data,f'{self.level2}/Statistics')
        scan_edges = self.getGroup(data,statistics,'scan_edges')


        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape
        #if 20 in feeds:
        #    nFeeds -= 1
        nScans = len(scan_edges)

        self.powerspectra = np.zeros((nFeeds, nBands, nChannels, nScans, self.nbins))
        self.freqspectra = np.zeros((nFeeds, nBands, nChannels, nScans, self.nbins))
        self.fnoise_fits = np.zeros((nFeeds, nBands, nChannels, nScans, 3))
        self.wnoise_auto = np.zeros((nFeeds, nBands, nChannels, nScans, 1))
        self.atmos = np.zeros((nFeeds, nBands, nScans, 3))
        self.atmos_errs = np.zeros((nFeeds, nBands, nScans, 3))

        self.filter_tods = [] # Store as a list of arrays, one for each "scan"
        self.filter_coefficients = np.zeros((nFeeds, nBands, nChannels, nScans, 1)) # Stores the per channel gradient of the  median filter
        self.atmos_coefficients = np.zeros((nFeeds, nBands, nChannels, nScans, 1)) # Stores the per channel gradient of the  median filter

        pbar = tqdm(total=(nFeeds*nBands*nScans),desc=self.name)
        for iscan,(start,end) in enumerate(scan_edges):
            local_filter_tods = np.zeros((nFeeds,nBands, end-start))
            for ifeed in range(nFeeds):
                if feeds[ifeed] == 20:
                    pbar.update(1)
                    continue
                for iband in range(nBands):
                    band_average = np.nanmean(tod[ifeed,iband,3:-3,start:end],axis=0)

                    select = self.getSourceMask(ra[ifeed,start:end],dec[ifeed,start:end],mjd[start:end])
                    _az = az[ifeed,start:end]
                    _el = el[ifeed,start:end]
                    atmos_filter,atmos,atmos_errs = self.FitAtmosAndGround(band_average,
                                                                           _az,
                                                                           _el,
                                                                           mask=select)
                    local_filter_tods[ifeed,iband,:] = self.median_filter(band_average-atmos_filter)[:band_average.size]

                    
                    self.atmos[ifeed,iband,iscan,:] = atmos
                    self.atmos_errs[ifeed,iband,iscan,:] = atmos_errs

                    #ps, nu, f_fits, w_auto = self.FitPowerSpectrum(band_average-atmos_filter-local_filter_tods[ifeed,iband,:])

                    for ichan in range(nChannels):
                        if np.nansum(tod[ifeed, iband, ichan,start:end]) == 0:
                            continue
                        # Check atmosphere coefficients
                        #try:
                        atmos_coeff,med_coeff,offset = self.coefficient_jointfit(tod[ifeed,iband,ichan,start:end], 
                                                                                 atmos_filter,
                                                                                 local_filter_tods[ifeed,iband,:], 
                                                                                 mask=select)

                        w_auto = stats.AutoRMS(tod[ifeed,iband,ichan,start:end])
                        self.wnoise_auto[ifeed,iband,ichan,iscan,:]  = w_auto
                        self.filter_coefficients[ifeed,iband,ichan,iscan,:] = med_coeff
                        self.atmos_coefficients[ifeed,iband,ichan,iscan,:]  = atmos_coeff
                        resid = tod[ifeed,iband,ichan,start:end]-atmos_filter*atmos_coeff-local_filter_tods[ifeed,iband,:]*med_coeff - offset

                        if self.isCalibrator: # Fill in data that is on source
                            Nfill = int(np.sum(~select))
                            resid[~select] = np.random.normal(size=Nfill,scale=w_auto)
                            
                        ps, nu, f_fits, w_auto = self.FitPowerSpectrum(resid)

                        self.powerspectra[ifeed,iband,ichan,iscan,:] = ps
                        self.freqspectra[ifeed,iband,ichan,iscan,:]  = nu
                        self.fnoise_fits[ifeed,iband,ichan,iscan,0]  = w_auto
                        self.fnoise_fits[ifeed,iband,ichan,iscan,1:] = f_fits

                    pbar.update(1)

            self.filter_tods += [local_filter_tods]

        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')


        self.source = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {self.source} - {comment}')

        if self.checkAllowedSources(data, self.source, self.allowed_sources):
            return data

        if any([s in self.source for s in ['jupiter','CygA','Jupiter','TauA','CasA']]):
            self.isCalibrator = True
        else:
            self.isCalibrator = False 

        if ('Sky nod' in comment) | ('Engineering Test' in comment):
            return data

        if ('level2/Statistics/fnoise_fits' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Measuring noise stats.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing noise stats to level 2 file ({fname})')
        self.write(data)
        if not isinstance(self.database,type(None)):
            self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def get_filter_coefficient(self,tod,median_filter):
        """
        Calculate the gradient between tod and filter
        """
        #print('TOD {}, FILTER {}'.format(tod.shape,median_filter.shape))
        return np.sum(tod*median_filter)/np.sum(median_filter**2)

    def coefficient_jointfit(self, _tod, _atmos, _med_filt,mask=None):
        """
        """
        if isinstance(mask,type(None)):
            mask = np.ones(_tod.size,dtype=bool)

        tod = _tod[mask]
        atmos=_atmos[mask]
        med_filt=_med_filt[mask]

        templates = np.ones((3,tod.size))
        templates[0,:] = atmos
        templates[1,:] = med_filt
        C = templates.dot(templates.T)
        z = templates.dot(tod[:,None])

        try:
            a = np.linalg.solve(C,z)
        except np.linalg.LinAlgError:
            a = np.zeros(templates.shape[0])
        return a.flatten()


    def median_filter(self,tod):
        """
        Calculate this AFTER removing the atmosphere.
        """
        if tod.size > 2*self.medfilt_stepsize:
            z = np.concatenate((tod[::-1],tod,tod[::-1]))
            filter_tod = np.array(medfilt.medfilt(z.astype(np.float64),np.int32(self.medfilt_stepsize)))[tod.size:2*tod.size]
        else:
            filter_tod = np.ones(tod.size)*np.nanmedian(tod)

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



    def FitAtmosAndGround(self,_tod,_az,_el,mask=None,niter=100):
        if isinstance(mask,type(None)):
            mask = np.ones(_tod.size,dtype=bool)

        tod =_tod[mask]
        az  =_az[mask]
        el  =_el[mask]
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

        # interpolate to mask
        tod_filter_all = np.zeros(_tod.size)
        tod_filter_all[mask] = tod_filter
        t = np.arange(_tod.size)
        tod_filter_all[~mask] = np.interp(t[~mask],t[mask],tod_filter)
        
        return tod_filter_all, fits, errs


    def RemoveAtmosphere(self, tod, el):
        """
        Remove 1/sin(E) relationship from TOD
        """
        A = 1/np.sin(el*np.pi/180) # Airmass
        pmdl = np.poly1d(np.polyfit(A, tod,1))
        return tod- pmdl(A), pmdl

    def write_database(self,data):
        """
        Write out the statistics to a common statistics database for easy access
        """
        
        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        feeds, feed_idx, feed_dict =self.getFeeds(data,'all')
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)


        if 'FnoiseStats' in grp:
            del grp['FnoiseStats']
        stats = grp.create_group('FnoiseStats')

        dnames = ['feeds','fnoise_fits','wnoise_auto', 'powerspectra','freqspectra',
                  'atmos','atmos_errors','filter_coefficients','atmos_coefficients']
        dsets = [feeds,self.fnoise_fits,self.wnoise_auto,self.powerspectra,self.freqspectra,
                 self.atmos,self.atmos_errs,self.filter_coefficients,self.atmos_coefficients]
        for (dname, dset) in zip(dnames, dsets):
            if dname in stats:
                del stats[dname]
            stats.create_dataset(dname,  data=dset)
        output.close()
            
    def write(self,data):
        """
        Write out fitted statistics to the level 2 file
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


class SkyDipStats(BaseClasses.DataStructure):
    """
    Takes level 2 files, bins and calibrates them for continuum analysis.
    Does not require scan_edges to run
    """

    def __init__(self,allowed_sources = ['fg','GField','Field','TauA','CasA','Jupiter','jupiter','Cyga'],
                 nbins=50, 
                 samplerate=50,
                 medfilt_stepsize=5000, 
                 poly_iter=100, 
                 database=None,
                 dipLo=42, 
                 dipHi=58):
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
        self.database=database
        self.allowed_sources = allowed_sources
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
        feat = np.log(data['level1/spectrometer/features'][...])/np.log(2)



        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nChannels, nSamples = tod.shape
        self.opacity = np.zeros((nFeeds, nBands, nChannels))
        self.opacity_err = np.zeros((nFeeds, nBands, nChannels))
        self.Tzen = np.zeros((nFeeds, nBands, nChannels))
        self.Tzen_err = np.zeros((nFeeds, nBands, nChannels))

        pbar = tqdm(total=((nFeeds-1)*nBands*nChannels))


        for ifeed in range(nFeeds):
            if feeds[ifeed] == 20:
                continue

            skydip_select = (el[ifeed]>self.dipLo) & (el[ifeed]<self.dipHi) & (feat == 8)
            pyplot.imshow(tod[0,0,:,:],aspect='auto')
            pyplot.show()
            for iband in range(nBands):

                for ichan in range(nChannels):
                    x = 1/(np.cos(el[ifeed,skydip_select[ifeed]]*(np.pi/180)))
                    y = tod[ifeed,iband,ichan,skydip_select[ifeed]]

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

                    pyplot.plot(x,y,',')
                    pyplot.plot(x,np.poly1d(avg)(x))
                    pyplot.show()
                    #assume Tatm=300K
                    self.opacity[ifeed,iband,ichan] = avg[1]/300#K
                    self.opacity_err[ifeed,iband,ichan] = std[1]/300#K
                    self.Tzen[ifeed,iband,ichan] = avg[0]
                    self.Tzen_err[ifeed,iband,ichan] = std[0]

                    pbar.update(1)

        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        source = self.getSource(data)
        comment = self.getComment(data)
        if self.checkAllowedSources(data, source, self.allowed_sources):
            return data

        if not 'Sky nod' in comment:
            return data

        self.run(data)

        # Want to ensure the data file is read/write
        self.setReadWrite(data)

        self.write(data)

        if not isinstance(self.database,type(None)):
            self.write_database(data)

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
    def write_database(self,data):
        """
        Write the skydip data to the database too 
        """
        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)

        if self.name in grp:
            lvl2 = grp[self.name]
        else:
            lvl2 = grp.create_group(self.name)

        lvl2.attrs['dipLo'] = self.dipLo
        lvl2.attrs['dipHi'] = self.dipHi

        dnames = ['opacity', 'opacity_err', 'Tzenith', 'Tzenith_err']
        dsets = [self.opacity, self.opacity_err, self.Tzen_err, self.Tzen_err]
        for (dname, dset) in zip(dnames, dsets):
            if dname in lvl2:
                del lvl2[dname]
            lvl2.create_dataset(dname,  data=dset)

class FeedFeedCorrelations(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self,level2='level2',database=None,**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'FeedFeedCorrelations'
        self.level2=level2
        self.database=database

    def __str__(self):
        return "Calculating feed feed correlations."

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]


        stats = data['level2/Statistics'] 
        medfilts = stats['FilterTod_Scan00'][...]
        stepsize = stats['FilterTod_Scan00'].attrs['medfilt_stepsize']
        N = int(medfilts.shape[-1]//stepsize * stepsize)
        medfilts = np.nanmean(np.reshape(medfilts[:,:,:N],(medfilts.shape[0],
                                                           medfilts.shape[1],
                                                           N//stepsize,
                                                           stepsize)),axis=-1)
        
        z = (medfilts[:,0,:] - np.nanmean(medfilts[:,0,:],axis=-1)[:,None])#/\
            # np.nanstd(medfilts[:,0,:],axis=-1)[:,None]
        C = z.dot(z.T)/medfilts.shape[-1]

        feeds,feedsidx,_ = self.getFeeds(data,'all')
        i = np.where((feeds == 8))[0][0]
        feedsidx = feedsidx[:i+1]
        C[feedsidx,feedsidx] = np.nan
        
        

        self.data_out = {'feed_feed_correlation':np.nanmean(C)}
        

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        source  = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if (f'level2/Statistics/{self.name}' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Calculating Sun distance.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing Sun distance to level 2 file ({fname})')
        self.write(data)
        if not isinstance(self.database,type(None)):
            self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

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

        if not self.name in statistics:
            grp = statistics.create_group(self.name)
        else:
            grp = statistics[self.name]

        
        for dname, dset in self.data_out.items():
            if dname in grp:
                del grp[dname]
            grp.create_dataset(dname,data=dset)

    def write_database(self,data):
        """
        Write out the statistics to a common statistics database for easy access
        """
        
        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)


        if  self.name in grp:
            del grp[self.name]
        stats = grp.create_group(self.name)


        for dname, dset in self.data_out.items():
            if dname in stats:
                del stats[dname]
            print(dname,dset)
            stats.create_dataset(dname,  data=dset)
        output.close()

class SunDistance(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self,level2='level2',database=None,**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'SunDistance'
        self.level2=level2
        self.database=database

    def __str__(self):
        return "Calculating sun distance."

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]

        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][0,:]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][0,:]
        mjd = data['level1/spectrometer/MJD'][:]

        self.distances = {k:np.zeros(az.size) for k in ['sun','moon']}

        for src, v in self.distances.items():
            s_az, s_el, s_ra, s_dec = Coordinates.sourcePosition(src, mjd, Coordinates.comap_longitude, Coordinates.comap_latitude)
            self.distances[src] = Coordinates.AngularSeperation(az,el,s_az,s_el)

        sources = list(self.distances.keys())
        for src in sources:
            self.distances[f'{src}_mean'] = np.array([np.mean(self.distances[src])])
        

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        source  = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if ('level2/Statistics/Distances' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Calculating Sun distance.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing Sun distance to level 2 file ({fname})')
        self.write(data)
        if not isinstance(self.database,type(None)):
            self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

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

        if not 'Distances' in statistics:
            distance_grp = statistics.create_group('Distances')
        else:
            distance_grp = statistics['Distances']

        
        for dname, dset in self.distances.items():
            if dname in distance_grp:
                del distance_grp[dname]
            distance_grp.create_dataset(dname,data=dset)
    def write_database(self,data):
        """
        Write out the statistics to a common statistics database for easy access
        """
        
        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)


        if  self.name in grp:
            del grp[self.name]
        stats = grp.create_group(self.name)


        for dname, dset in self.distances.items():
            if not 'mean' in dname:
                continue
            if dname in stats:
                del stats[dname]
            stats.create_dataset(dname,  data=dset)
        output.close()


class WindSpeed(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self,level2='level2',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'WindSpeed'
        self.level2=level2

    def __str__(self):
        return "Calculating sun distance."

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]

        az  = data['level1/spectrometer/pixel_pointing/pixel_az'][0,:]
        el  = data['level1/spectrometer/pixel_pointing/pixel_el'][0,:]
        mjd = data['level1/spectrometer/MJD'][:]

        self.distances = {k:np.zeros(az.size) for k in ['sun','moon']}

        for src, v in self.distances.items():
            s_az, s_el, s_ra, s_dec = Coordinates.sourcePosition(src, mjd, Coordinates.comap_longitude, Coordinates.comap_latitude)
            self.distances[src] = Coordinates.AngularSeperation(az,el,s_az,s_el)

        sources = list(self.distances.keys())
        for src in sources:
            self.distances[f'{src}_mean'] = np.array([np.mean(self.distances[src])])
        

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        source  = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if ('level2/Statistics/Distances' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Calculating Sun distance.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing Sun distance to level 2 file ({fname})')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

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

        if not 'Distances' in statistics:
            distance_grp = statistics.create_group('Distances')
        else:
            distance_grp = lvl2['distance_grp']

        
        for dname, dset in self.distances.items():
            if dname in distance_grp:
                del distance_grp[dname]
            distance_grp.create_dataset(dname,data=dset)


class SpikeFlags(BaseClasses.DataStructure):
    """
    Search TODs for transient spikes
    """

    def __init__(self,level2='level2',database=None,**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'Spikes'
        self.level2=level2
        self.database=database

    def __str__(self):
        return "Calculating spike mask"

    def remove_filter(self,data,feedtod,ifeed,level2='level2'):

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

                    mdl = AtmosGroundModel(atmos[iband,iscan],az[start:end],el[start:end]) *\
                          atmos_coefficient[iband,ichan,iscan,0]
                    mdl += median_filter[iband,:N] * medfilt_coefficient[iband,ichan,iscan,0]
                    feedtod[iband,ichan,start:end] -= mdl
                    feedtod[iband,ichan,start:end] -= np.nanmedian(feedtod[iband,ichan,start:end])
        return np.nanmean(feedtod,axis=(0,1)), mask

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]

        self.feeds, self.feedidx,_ = self.getFeeds(data,'all')
        print(data.keys())
        tod = data['level2/averaged_tod']
        todall = np.zeros((tod.shape[0],tod.shape[-1]))
        for ifeed,feed in enumerate(self.feeds):
            if feed > 8:
                continue
            if feed == 20:
                continue
            todall[ifeed], mask = self.remove_filter(data,tod[ifeed],ifeed,level2='level2')

        idx = np.argmin(np.abs(self.feeds-8))
        z = np.nanmedian(todall[:idx,mask],axis=0)
        peaks, properties = find_peaks(z, prominence=0.5,width=[1,200])

        N = len(properties['left_ips'])
        self.output={'mask':np.ones(tod.shape[-1],dtype=bool),
                     'left':np.zeros(N,dtype=int),
                     'right':np.zeros(N,dtype=int),
                     'width':np.zeros(N,dtype=int)}

        mask_idx = np.where(mask)[0]
        for i,(left,right,w) in enumerate(zip(properties['left_ips'],properties['right_ips'],properties['widths'])):
            lo = mask_idx[int(left-w)]
            hi = mask_idx[int(right+w)]
            self.output['mask'][lo:hi] = False
            self.output['left'][i] = lo
            self.output['right'][i]= hi
            self.output['width'][i]= int(w*3)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        source  = self.getSource(data)
        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {source} - {comment}')

        if ('level2/Statistics/{self.name}' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Calculating Spike Mask.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing Spike Mask to level 2 file ({fname})')
        self.write(data)
        if not isinstance(self.database,type(None)):
            self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def write(self,data):
        """
        """
        fname = data.filename.split('/')[-1]

        if not self.level2 in data:
            return
        lvl2 = data[self.level2]
        if not 'Statistics' in lvl2:
            statistics = lvl2.create_group('Statistics')
        else:
            statistics = lvl2['Statistics']

        if not self.name in statistics:
            grp = statistics.create_group(self.name)
        else:
            grp = statistics[self.name]

        
        for dname, dset in self.output.items():
            if dname in grp:
                del grp[dname]
            grp.create_dataset(dname,data=dset)

    def write_database(self,data):
        """
        """
        
        if not os.path.exists(self.database):
            db = FileTools.safe_hdf5_open(self.database,'w')
        else:
            db = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        if obsid in db:
            grp = db[obsid]
        else:
            grp = db.create_group(obsid)


        if  self.name in grp:
            del grp[self.name]
        stats = grp.create_group(self.name)


        for dname in ['left','right','width']:
            dset = self.output[dname]
            if dname in stats:
                del stats[dname]
            stats.create_dataset(dname,  data=dset)
        db.close()
