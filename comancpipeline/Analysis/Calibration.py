import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types
from os import listdir, getcwd
from os.path import isfile, join
from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import pandas as pd
#from mpi4py import MPI 
import os
#comm = MPI.COMM_WORLD

class CreateLevel2Cont(SourceFitting.FitSource):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1, average_width=512,calvanedir='AncillaryData/CalVanes'):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """

        self.feeds = feeds
        
        self.output_dir = output_dir

        self.nworkers = int(nworkers)
        self.average_width = int(average_width)
        self.calvanedir = calvanedir


    def run(self,data):
        """
        Sets up feeds that are needed to be called,
        grabs the pointer to the time ordered data,
        and calls the averaging routine in SourceFitting.FitSource.average(*args)

        """
        # Get data structures we need
        alltod = data['spectrometer/tod']

        # Setup feed indexing
        # self.feeds : feed horn ID (in array indexing, only chosen feeds)
        # self.feedlist : all feed IDs in data file (in lvl1 indexing)
        # self.feeddict : map between feed ID and feed array index in lvl1
        if self.feeds == 'all':
            feeds = data['spectrometer/feeds'][:]
        else:
            if (not isinstance(self.feeds,list)) & (not isinstance(self.feeds,np.ndarray)) :
                self.feeds = [int(self.feeds)]
                feeds = self.feeds
            else:
                feeds = [int(f) for f in self.feeds]

        self.feeds = feeds
        self.feedlist = data['spectrometer/feeds'][:]
        self.feeddict = {feedid:feedindex for feedindex, feedid in enumerate(self.feedlist)}


        # Setup output file here, we will have to write in sections.
        prefix = data.filename.split('/')[-1].split('.hd5')[0]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.outfilename = '{}/{}_Level2Cont.hd5'.format(self.output_dir,prefix)
        if os.path.isfile(self.outfilename):
            os.remove(self.outfilename)


        # Opening file here to write out data bit by bit
        self.outfile = h5py.File(self.outfilename)
        lvl2 = self.outfile.create_group('level2')
        avg_tod_shape = (alltod.shape[0], alltod.shape[1], alltod.shape[2]//self.average_width, alltod.shape[3])
        tod = lvl2.create_dataset('averaged_tod',avg_tod_shape, dtype=alltod.dtype)

        # self.average is inherited from SourceFitting.FitSource to keep both
        # methods consistent.
        self.average(data.filename,data,alltod, tod)


    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        self.run(data)
        self.write(data)

        # Now change to the level2 file for all future analyses.
        data.close()
        data = self.outfile

        return self.outfile

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """        


        #lvl2 = outfile.create_group('level2')
        #lvl2.create_dataset('averaged_tod',data=self.downsampled_tod)
        self.outfile['level1'] = h5py.ExternalLink(data.filename,'/')

class CoordOffset(DataStructure):
    """
    Checks if the pointing has been generated and generates it if not.
    """
    
    def __init__(self, longitude=-118.2941, latitude=37.2314,offset=0, force=True):
        super().__init__()

        if isinstance(force, type(str)):
            self.force = (force.lower() == 'true')
        else:
            self.force = force

        # stores and write the lon/lat of the telescope to the output file
        # default is the COMAP telescope geo coordinates
        self.longitude = longitude
        self.latitude = latitude
        self.offset = offset / 1000. / 24./ 3600.

        self.fields = ['spectrometer/pixel_pointing/{}'.format(v) for v in ['pixel_az','pixel_el', 'pixel_ra', 'pixel_dec']]

        self.focalPlane = FocalPlane()

    def __str__(self):
        return "Adding {}ms offset to encoder timing.".format(self.offset*1000.*24*3600.)

    def run(self,data):
            
        self.longitude = data.getdset('hk/antenna0/tracker/siteActual')[0,0]/(60.*60.*1000.)
        self.latitude  = data.getdset('hk/antenna0/tracker/siteActual')[0,1]/(60.*60.*1000.)
        nHorn, nSB, nChan, nSample = data.getdset('spectrometer/tod').shape

        az = data.getdset(self.fields[0])
        el = data.getdset(self.fields[1])
        ra = data.getdset(self.fields[2])
        dec= data.getdset(self.fields[3])
        mjd   = data.getdset('spectrometer/MJD')

        if data.splitType == Types._HORNS_:
            splitFull = data.fullFieldLengths[data.splitType]
            lo, hi    = data.getDataRange(splitFull)
        else:
            lo, hi = 0, nHorn

        for ilocal, i in enumerate(range(lo,hi)):

            def interpC(mjd,pos):
                azmdl = interp1d(mjd,pos, bounds_error=False)
                aznew = azmdl(mjd + self.offset)
                
                badVals = np.where((np.isnan(aznew)))[0]
                if len(badVals) > 0:
                    if np.max(badVals) < az.shape[1]/2.:
                        aznew[badVals] = aznew[max(badVals)+1]
                    else:
                        aznew[badVals] = aznew[min(badVals)-1]
                    return aznew
                else:
                    return aznew

            aznew = interpC(mjd, az[i,:])
            elnew = interpC(mjd, el[i,:])
            # fill in the ra/dec fields

            _ra,_dec = Coordinates.h2e(aznew,
                                       elnew,
                                       mjd, 
                                       self.longitude, 
                                       self.latitude)

            _ra, _dec = Coordinates.precess(_ra,_dec,mjd)
            ra[i,:] = _ra
            dec[i,:] = _dec

class FixTiming(CoordOffset):

    def interpAzEl(self, cval, cmjd, todmjd):
        mdl = interp1d(cmjd, cval, bounds_error=False, fill_value=np.nan)
        posout = mdl(todmjd)
        badVals = np.where((np.isnan(posout)))[0]
        if len(badVals) > 0:
            if np.max(badVals) < todmjd.size/2.:
                posout[badVals] = posout[max(badVals)+1]
            else:
                posout[badVals] = posout[min(badVals)-1]
            return posout
        else:
            return posout

    def run(self, data):
        self.longitude = data.getdset('hk/antenna0/tracker/siteActual')[0,0]/(60.*60.*1000.)
        self.latitude  = data.getdset('hk/antenna0/tracker/siteActual')[0,1]/(60.*60.*1000.)
        self.focalPlane = FocalPlane()

        nHorn, nSB, nChan, nSample = data.getdset('spectrometer/tod').shape

        az = data.getdset(self.fields[0])
        el = data.getdset(self.fields[1])
        ra = data.getdset(self.fields[2])
        dec= data.getdset(self.fields[3])
        mjd   = data.getdset('spectrometer/MJD')
        mjd = mjd[0] + np.arange(mjd.size)*self.offset # new timing offsets
        azact = data.getdset('pointing/azActual')
        elact = data.getdset('pointing/elActual')
        pmjd  = data.getdset('pointing/MJD')

        azcen = self.interpAzEl(azact,
                                pmjd,
                                mjd)
        elcen = self.interpAzEl(elact,
                                pmjd,
                                mjd)

        tod = data.getdset('spectrometer/tod')

        if data.splitType == Types._HORNS_:
            splitFull = data.fullFieldLengths[data.splitType]
            lo, hi    = data.getDataRange(splitFull)
        else:
            lo, hi = 0, nHorn

        for ilocal, i in enumerate(range(lo,hi)):
            el[i,:] = elcen+self.focalPlane.offsets[i][1] - self.focalPlane.eloff
            az[i,:] = azcen+self.focalPlane.offsets[i][0]/np.cos(el[i,:]*np.pi/180.) - self.focalPlane.azoff

            # fill in the ra/dec fields
            _ra,_dec = Coordinates.h2e(az[i,:],
                                       el[i,:],
                                       mjd, 
                                       self.longitude, 
                                       self.latitude)

            _ra, _dec = Coordinates.precess(_ra,_dec,mjd)
            ra[i,:]  = _ra
            dec[i,:] = _dec

class RefractionCorrection(DataStructure):
    """
    Use internal COMAP registers to correct for refraction in elevation
    """
    
    def __init__(self):
        super().__init__()

        self.fields = ['spectrometer/pixel_pointing/{}'.format(v) for v in ['pixel_az','pixel_el', 'pixel_ra', 'pixel_dec']]
        self.focalPlane = FocalPlane()

    def __str__(self):
        return "Correcting for refraction term.".format()

    def interp(self, cval, cmjd, bmjd):
        mdl = interp1d(cmjd, cval, bounds_error=False, fill_value=np.nan)
        bval = mdl(bmjd)
        badVals = np.where((np.isnan(bval)))[0]
        if len(badVals) > 0:
            if np.max(badVals) < bmjd.shape[0]/2.:
                bval[badVals] = bval[max(badVals)+1]
            else:
                bval[badVals] = bval[min(badVals)-1]
        bval[np.isnan(bval)] = 0
        return bval

    def run(self,data):
        deg2mas = 1000. * 3600.
        refraction = data.getdset('hk/antenna0/deTracker/refraction')[:,-1] / deg2mas
        refracUTC = data.getdset('hk/antenna0/deTracker/utc')

        el = data.getdset(self.fields[1])
        mjd   = data.getdset('spectrometer/MJD')
        refraction = self.interp(refraction, refracUTC, mjd) # interpolate to TOD time
        # correct elevation
        el += refraction[np.newaxis,:] # do not correct for each horn, we could do that later?
        print('ELEVATION', np.sum(refraction))


class FillPointing(DataStructure):
    """
    Checks if the pointing has been generated and generates it if not.
    """
    
    def __init__(self, longitude=-118.2941, latitude=37.2314, force=True):
        super().__init__()

        if isinstance(force, type(str)):
            self.force = (force.lower() == 'true')
        else:
            self.force = force

        # stores and write the lon/lat of the telescope to the output file
        # default is the COMAP telescope geo coordinates
        self.longitude = longitude
        self.latitude = latitude

        self.focalPlane = FocalPlane()

    def __str__(self):
        return "Filling any missing pointing information"

    def missingFields(self, data):
        
        missingFields = []
        checkFields = ['spectrometer/pixel_pointing/{}'.format(v) for v in ['pixel_az','pixel_el', 'pixel_ra', 'pixel_dec']]
        if self.force:
            return checkFields

        for field in checkFields:
            if not field in data.data:
                missingFields += [field]

        if len(missingFields) == 0:
            return None
        else:
            return missingFields
                  
    def interpAzEl(self, cval, cmjd, todmjd):
        mdl = interp1d(cmjd, cval, bounds_error=False, fill_value=np.nan)
        return mdl(todmjd)

    def run(self,data):
        missingFields = self.missingFields(data)
        if not isinstance(missingFields, type(None)):
            
            if 'spectrometer/tod' in data.dsets:
                nHorn, nSB, nChan, nSample = data.dsets['spectrometer/tod'].shape
            else:
                nHorn, nSB, nChan, nSample = data.data['spectrometer/tod'].shape
            
            azact = data.getdset('pointing/azActual')
            elact = data.getdset('pointing/elActual')
            mjd   = data.getdset('spectrometer/MJD')
            pmjd  = data.getdset('pointing/MJD')

            azval = self.interpAzEl(data.getdset('pointing/azActual'),
                                    data.getdset('pointing/MJD'),
                                    data.getdset('spectrometer/MJD'))
            elval = self.interpAzEl(data.getdset('pointing/elActual'),
                                    data.getdset('pointing/MJD'),
                                    data.getdset('spectrometer/MJD'))
            #print(mjd)
            #print((np.max(mjd)-np.min(mjd))*24*60, (np.max(pmjd)-np.min(pmjd))*24*60)
            #pyplot.plot(azval, elval)
            #pyplot.show()
            
            #az,el, ra, dec = [np.zeros(( nHorn, nSample)) for i in range(4)]
            missingFields = np.sort(np.array(missingFields))
            for field in np.sort(missingFields):
                if not field in data.data:
                    data.data.create_dataset(field, (nHorn, nSample), dtype='f')

            if data.splitType == Types._HORNS_:
                splitFull = data.fullFieldLengths[data.splitType]
                lo, hi    = data.getDataRange(splitFull)
            else:
                lo, hi = 0, nHorn

            for ilocal, i in enumerate(range(lo,hi)):
                data.data['spectrometer/pixel_pointing/pixel_el'][i,:] = elval+self.focalPlane.offsets[i][1] - self.focalPlane.eloff
                data.data['spectrometer/pixel_pointing/pixel_az'][i,:] = azval+self.focalPlane.offsets[i][0]/np.cos(data.data['spectrometer/pixel_pointing/pixel_el'][i,:]*np.pi/180.) - self.focalPlane.azoff
                    
                # fill in the ra/dec fields
                data.data['spectrometer/pixel_pointing/pixel_ra'][i,:],data.data['spectrometer/pixel_pointing/pixel_dec'][i,:] = Coordinates.h2e(data.data['spectrometer/pixel_pointing/pixel_az'][i,:],
                                                                                                                                                 data.data['spectrometer/pixel_pointing/pixel_el'][i,:],
                                                                                                                                                 data.getdset('spectrometer/MJD'), 
                                                                                                                                                 self.longitude, 
                                                                                                                                                 self.latitude)
        


        data.setAttr('comap', 'cal_fillpointing', True)


class DownSampleFrequency(DataStructure):
    """
    Downsample data in frequency.
    THIS IS A SPECIAL FUNCTION: It cannot be called with MPI due to data volume and IO issues.
    """
    def __init__(self, out_dir='', factor=16):
        super().__init__()
        self.mode = 'r' # down sampling is transfered to a new file.
        self.out_dir = out_dir
        self.factor = int(factor)

    def __str__(self):
        return "Downsampling in frequency by a factor of {}".format(self.factor)

    def createDatasets(self, din):
        """
        Copy all groups/datasets/attributes that are not downsampled (e.g., those without a frequency dependency) to the new downsampled file.
        """
        din.data.copy('comap/', self.dout)
        for key, attr in din.data['comap'].attrs.items():
            self.dout['comap'].attrs[key] = attr

        #hk    = self.dout.create_group('hk')
        #hkgroups = ['deTracker', 'drivenode', 'env', 'saddlebag', 'vane']
        #for hkgroup in hkgroups:
        #    din.data.copy('hk/{}'.format(hkgroup), hk)

        din.data.copy('pointing/', self.dout)

        din.data.copy('weather/', self.dout)

        spectrometer = self.dout.create_group('spectrometer')
        specsets = ['MJD', 'band_average', 'bands', 'feeds']
        for specset in specsets:
            spectrometer.create_dataset('{}'.format(specset), 
                                        data=din.data['spectrometer/{}'.format(specset)])

        din.data.copy('spectrometer/pixel_pointing', spectrometer)
    

    def run(self,data):
        
        for field, desc in Types._COMAPDATA_.items():
            if field == 'spectrometer/tod':
                continue

            dset = data.getdset(field)
            try:
                meanAxis = np.where((np.array(desc) == Types._FREQUENCY_))[0][0]
            except IndexError:
                continue

            dsetShape = dset.shape
            if not (meanAxis == len(dsetShape)): # swap order so frequency is last
                dshape = np.arange(len(dset.shape)).astype(int)
                transShape = np.arange(len(dset.shape)).astype(int)
                transShape[-1] = dshape[meanAxis]
                transShape[meanAxis] = dshape[-1]
                dset = np.transpose(dset, transShape)

            meanShape = list(dsetShape) + [self.factor]
            meanShape[-2] = meanShape[-2]//self.factor
            dset = np.nanmean(np.reshape(dset, meanShape),axis=-1)
            
            if not (meanAxis == len(dsetShape)): # swap back to original ordering
                dshape = np.arange(len(dset.shape)).astype(int)
                transShape = np.arange(len(dset.shape)).astype(int)
                transShape[-1] = dshape[meanAxis]
                transShape[meanAxis] = dshape[-1]
                dset = np.transpose(dset, transShape)
            data.resizedset(field, dset)

        # field = 'spectrometer/frequency'
        # freq = data.getdset(field) 
        # freq = np.mean(freq.reshape((freq.shape[0], freq.shape[1]//self.factor, self.factor)),axis=-1)
        # data.resizedset(field, freq)

        # field = 'spectrometer/time_average'
        # time_average = data.getdset(field)
        # time_average = time_average.reshape((time_average.shape[0], time_average.shape[1],
        #                       time_average.shape[2]//self.factor, 
        #                       self.factor))
        # time_average = np.mean(time_average, axis=-1)
        # data.resizedset(field, time_average)

        # can't read in all of the TOD usually...
        field = 'spectrometer/tod'
        tod = data.getdset(field)
        todOut = np.zeros((tod.shape[0], tod.shape[1], tod.shape[2]//self.factor, tod.shape[3]))
        # ...read in one horn at a time
        nHorns, nSBs, nChans, nSamps = tod.shape
        #import time
        for i in range(nHorns):
            # need frequency to be the last index...
            for j in range(nSBs):
                #start=time.time()
                temp = np.transpose(tod[i,j,:,:], (1,0))

                temp = np.reshape(temp, (temp.shape[0],
                                         temp.shape[1]//self.factor, 
                                         self.factor))
                todOut[i,j,:,:] = np.transpose(np.nanmean(temp,axis=-1),(1,0))
                #print(i,j,'Time taken {}'.format(time.time()-start))

        #        pyplot.plot(np.nanmean(todOut[i,j,:,:],axis=-1))
        #print(tod.shape,flush=True)

        #pyplot.show()
        Types._SHAPECHANGES_[Types._FREQUENCY_] = True
        data.resizedset(field, todOut)
        data.setAttr('comap', 'cal_downsampfreq', True)
        data.setAttr('comap', 'cal_downsampfreqfactor', self.factor)

class AmbLoadCal(DataStructure):
    """
    Interpolate Ambient Measurements
    """
    def __init__(self, amb_dir='AmbientLoads/', force=False):
        super().__init__()
        self.amb_dir = amb_dir
        self.force = False
        self.grp = 'AMBIENTLOADS'
        
    def __str__(self):
        return "Applying ambient load calibration"

    def getNearestGain(self, data):
        meanmjd = np.nanmean(data.getdset('pointing/MJD'))

        # First get the mean MJD of the data
        obsids = np.array([int(f.split('-')[-5]) for f in listdir(self.amb_dir) if isfile(join(self.amb_dir, f)) if ('hd5' in f) | ('hdf5' in f) ])
        obsid  = int(data.filename.split('-')[-5])

        gainfiles = [f for f in listdir(self.amb_dir) if isfile(join(self.amb_dir, f)) if ('hd5' in f) | ('hdf5' in f) ]

        # find the 10 closest obsids:
        argmins = np.argsort((obsids - obsid)**2)[:10]
        gainmjd = np.zeros(len(argmins))
        gainel = np.zeros(len(argmins))
        for i,j in enumerate(argmins):
            if comm.size > 1:
                g = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[j]),'r', driver='mpio', comm=comm)
            else:
                g = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[j]),'r')

            if self.grp in g:
                gainmjd[i] = g['{}/MJD'.format(self.grp)][0]
                gainel[i]  = g['{}/EL'.format(self.grp)][0]
            g.close()

        Amass = 1./np.sin(gainel*np.pi/180.)

        if all((Amass == np.inf)):
            AmassRange = np.arange(gainmjd.size).astype(int)
        else:
            AmassRange = np.where((Amass < 2))[0] # almost zenith

        if len(AmassRange) == 0:
            AmassRange = np.arange(gainmjd.size).astype(int)

        # Now find the nearest mjd to obs
        minmjd = np.argmin((gainmjd[AmassRange] - meanmjd)**2)
        if comm.size > 1:
            g = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[AmassRange[minmjd]]),'r', driver='mpio', comm=comm)
        else:
            g = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[AmassRange[minmjd]]),'r')

        self.Gain = g['{}/Gain'.format(self.grp)][...]
        self.GainFreq = g['{}/Frequency'.format(self.grp)][...]
        g.close()

        # DOWNSAMPLE TO THE CORRECT FREQUENCY BINNING
        freq = data.getdset('spectrometer/frequency')
        if data.splitType == Types._FREQUENCY_:
            nFreqs = data.fullFieldLengths[Types._FREQUENCY_]
        else:
            nFreqs = freq.shape[1]


        if self.Gain.shape[2] != nFreqs:
            if ('comap' in data.data) and ('cal_downsampfreqfactor' in data.data['comap'].attrs):
                self.factor = data.data['comap'].attrs['cal_downsampfreqfactor']
            elif ('comap' in data.attrs) and ('cal_downsampfreqfactor' in data.attrs['comap']):
                self.factor = data.attrs['comap']['cal_downsampfreqfactor']
            else:
                self.factor = int(self.Gain.shape[2]//nFreqs)
                print('Warning: No down sample factor found, estimating to be: {}'.format(self.factor),flush=True)

            self.Gain = np.mean(np.reshape(self.Gain, (self.Gain.shape[0],
                                                       self.Gain.shape[1], 
                                                       self.Gain.shape[2]//self.factor, 
                                                       self.factor)),axis=-1)


        # CROP TO THE DESIRED SELECT
        desc = [Types._HORNS_, Types._SIDEBANDS_, Types._FREQUENCY_]
        self.Gain = data.resizeextra(self.Gain, desc)
        
        return self.Gain, self.GainFreq

    def run(self, data):
        # don't want to calibrate multiple times!
        #if ('comap' in data.data):
        #    if (not isinstance(data.data['comap'].attrs.get('cal_ambload'), type(None))) and (not self.force):
        #        print('2 {} is already ambient load calibrated'.format(data.filename.split('/')[-1]))
        #        return None

        self.Gain, self.GainFreq = self.getNearestGain(data)

        tod = data.getdset('spectrometer/tod')
        tod /= self.Gain[...,np.newaxis]


        data.setAttr('comap', 'cal_ambload', True)


from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

class NoColdError(Exception):
    pass
class NoHotError(Exception):
    pass

class AmbientLoad2Gain(DataStructure):
    """
    Calculate AmbientLoad temperature
    """
    def __init__(self, output_dir='AmbientLoads/', overwrite=True):
        super().__init__()
        self.output_dir = output_dir
        self.overwrite= overwrite

        self.suffix = '_AmbLoad'

        self.elLim = 5.
        self.minSamples = 200

        self.tCold = 2.73 # K, cmb
        self.tHotOffset = 273.15 # celcius -> kelvin

    def __str__(self):
        return "Calculating Tsys and Gain from Ambient Load observation."

    def findHotCold(self, mtod):
        """
        Find the hot/cold sections of the ambient load scan
        """

        nSamps = mtod.size
        # Assuming there is a big step we take mid-value
        midVal = (np.nanmax(mtod)+np.nanmin(mtod))/2.

        # Sort the tod...
        mtodSort = np.sort(mtod)
        mtodArgsort = np.argsort(mtod)

        # Find the whitenoise rms
        rms = np.nanstd(mtod[:(nSamps//2)*2:2]-mtod[1:(nSamps//2)*2:2])/np.sqrt(2)

        # Where is it hot? where is it cold?
        groupHot  = (mtodSort-midVal) > rms*5
        groupCold = (mtodSort-midVal) < rms*5

        # Now return unsorted data 
        hotTod = mtod[(mtodArgsort[groupHot])]
        X =  np.abs((hotTod - np.median(hotTod))/rms) < 1
        if np.sum(X) == 0:
            raise NoHotError('No hot load data found')

        snips = int(np.min(np.where(X)[0])), int(np.max(np.where(X)[0]))
        idHot = (mtodArgsort[groupHot])[X]

        coldTod = mtod[(mtodArgsort[groupCold])]


        X =  np.abs((coldTod - np.median(coldTod))/rms) < 1
        if np.sum(X) == 0:
            raise NoColdError('No cold load data found')

        snips = int(np.min(np.where(X)[0])), int(np.max(np.where(X)[0]))
        idCold = (mtodArgsort[groupCold])[X]
        idCold = np.sort(idCold)

        # Just find cold AFTER calvane
        diffCold = idCold[1:] - idCold[:-1]
        jumps = np.where((diffCold > 50))[0]

        if len(jumps) == 0:
            snip = 0
        else:
            snip = int(jumps[0])
        
        idHot = np.sort(idHot)
        idCold = np.sort(idCold[snip:])
        return idHot[50:-50], idCold[50:-50]

    def TsysRMS(self,tod):
        N = (tod.shape[0]//2)*2
        diff = tod[1:N:2,:] - tod[:N:2,:]
        rms = np.nanstd(diff,axis=0)/np.sqrt(2)
        return rms

    def run(self,data):
        """
        Calculate the cal vane calibration factors for observations after
        June 14 2019 (MJD: 58635) after which corrections to the calvane features
        were made.

        """
        # Read in data that is required:
        freq = data['spectrometer/frequency'][...]
        mjd  = data['spectrometer/MJD'][:] # data.data['spectrometer/MJD'][:]
        el  = data['pointing/elActual'][:]        
        self.feeds = data['spectrometer/feeds'][:]
        self.mjd = np.mean(mjd)
        self.elevation = np.nanmedian(el)

        # Keep TOD as a file object to avoid reading it all in
        tod  = data['spectrometer/tod'] 
        btod  = data['spectrometer/band_average'] 
        nHorns, nSBs, nChan, nSamps = tod.shape



        # Setup for calculating the calibration factors, interpolate temperatures
        tHot  = data['hk/antenna0/vane/Tvane'][:]/100. + self.tHotOffset 
        hkMJD = data['hk/antenna0/vane/utc'][:]
        tHot = np.nanmean(tHot)

        # Need to count how many calvane events occur, look for features 2**13
        if mjd[0] > 58648: # The cal vane feature bit can be trusted after 14 June 2019
            features = np.floor(np.log(data['spectrometer/features'][:])/np.log(2)).astype(int)
            justDiodes = np.where((features == 13))[0]
        else: # Must use cal vane angle to calculate the diode positions
            angles = np.interp(mjd,hkMJD, data['hk/antenna0/vane/angle'][:])
            justDiodes = np.where((angles < 21000))[0]

        if len(justDiodes) == 0:
            self.nodata = True
            return 
        else:
            self.nodata = False

        nSamples2 = int((justDiodes.size//2)*2)
        diffFeatures = justDiodes[1:] - justDiodes[:-1]
        # Anywhere where diffFeatures > 60 seconds must be another event
        timeDiodes = int(60*50) # seconds * sample_rate
        events = np.where((diffFeatures > timeDiodes))[0]

        nDiodes = len(events) + 1

        # Calculate the cal vane start and end positions
        diodePositions = []
        for i in range(nDiodes):
            if i == 0:
                low = justDiodes[0]
            else:
                low = justDiodes[events[i-1]+1]
            if i < nDiodes-1:
                high = justDiodes[events[i]]
            else:
                high = justDiodes[-1]
            diodePositions += [[low,high]]


        self.deltaTs = np.zeros(nDiodes)
        # Create output containers
        self.Tsys = np.zeros((nDiodes, nHorns, nSBs, nChan))
        self.Gain    = self.Tsys*0.
        self.RMS    = self.Tsys*0.

        # Now loop over each event:
        for horn, feedid in enumerate(tqdm(self.feeds)):

            for diode_event, (start, end) in enumerate(diodePositions):
                # Get the mean time of the event
                if horn == 0:
                    self.deltaTs[diode_event] = int(diode_event) #int((np.mean(mjd[start:end])-mjd[0])*24*3600)

                tod_slice = tod[horn,:,:,start:end]
                btod_slice = btod[horn,:,start:end]


                #try:
                idHot, idCold = self.findHotCold(btod_slice[0,:])
                #except (NoHotError, NoColdError):
                #    continue

                hotcold = np.zeros(tod_slice.shape[-1])
                hotcold[idHot] = 2
                hotcold[idCold] = 1

                time= np.arange(tod_slice.shape[-1])
                for sb in range(nSBs):
                    vHot = np.nanmedian(tod_slice[sb,:,idHot],axis=0)
                    vCold= np.nanmedian(tod_slice[sb,:,idCold],axis=0)
                    Y = vHot/vCold
                    self.Tsys[diode_event,horn,sb,:] = ((tHot - self.tCold)/(Y - 1.) ) - self.tCold
                    self.Gain[diode_event,horn,sb,:] = ((vHot - vCold)/(tHot - self.tCold))

                    #t = np.arange(btod_slice.shape[-1])
                    #pyplot.plot(tod_slice[sb,-30,:])
                    #pyplot.plot(t[hotcold == 2], tod_slice[sb,-30,hotcold==2])
                    #pyplot.plot(t[hotcold == 1], tod_slice[sb,-30,hotcold==1])
                    #pyplot.title(horn)
                    #pyplot.show()

                    tod_slice[sb,:,:] /= self.Gain[diode_event,horn,sb,:,np.newaxis]
                    self.RMS[diode_event,horn,sb,:] = self.TsysRMS(tod_slice[sb,:,idCold])
            
    def write(self,data):
        """
        Write the Tsys, Gain and RMS to a pandas data frame for each hdf5 file.
        """        
        if self.nodata:
            return

        nHorns, nSBs, nChan, nSamps = data['spectrometer/tod'].shape

        # Structure:
        #                                    Frequency (GHz)
        # Date, DeltaT, Mode, Horn, SideBand     
        freq = data['spectrometer/frequency'][...]
        startDate = Types.Filename2DateTime(data.filename)

        # Reorder the data
        horns = data['spectrometer/feeds'][:]
        sidebands = np.arange(4).astype(int)
        modes = ['Tsys', 'Gain', 'RMS']
        iterables = [[startDate], self.deltaTs, modes, horns, sidebands]
        names = ['Date', 'DeltaT','Mode','Horn','Sideband']
        index = pd.MultiIndex.from_product(iterables, names=names)

        df = pd.DataFrame(index=index, columns=np.arange(nChan))
        idx = pd.IndexSlice
        
        for i in range(self.Tsys.shape[0]):
            df.loc(axis=0)[idx[:,i,'Tsys',:,:]] = np.reshape(self.Tsys[i,...], (nHorns*nSBs, nChan))
            df.loc(axis=0)[idx[:,i,'Gain',:,:]] = np.reshape(self.Gain[i,...], (nHorns*nSBs, nChan))
            df.loc(axis=0)[idx[:,i,'RMS',:,:]]  = np.reshape(self.RMS[i,...],  (nHorns*nSBs, nChan))

        # save the dataframe
        # outfilename = '{}/DataFrame_TsysGainRMS.pkl'.format(self.output_dir)
        # if os.path.exists(outfilename):
        #     dfAll = df.read_pickle(outfilname)
        #     dfAll = dfAll.append(df)
        # else:
        #     dfAll = df
        prefix = data.filename.split('/')[-1].split('.hd5')[0]
        df.to_pickle('{}/{}_TsysGainRMS.pkl'.format(self.output_dir,prefix))

        

from comancpipeline.Tools import CaliModels

class Jupiter2Gain(DataStructure):
    """
    Expects Jupiter source fitting to have been performed prior
    """

    def run(self, data):
        Pout = data.getextra('JupiterFits/Parameters')
        freq = data.getextra('JupiterFits/frequency')
        mjd = data.getdset('spectrometer/MJD')
        el  = data.getdset('spectrometer/pixel_pointing/pixel_el')

        # Jupiter flux
        Sjup, dist =  CaliModels.JupiterFlux(freq, np.array([np.mean(mjd)]))

        # Flux scale conversion
        self.G = Pout[:,:,:,0]/Sjup[np.newaxis,...]
        
        JUPITERCAL = 'JUPITERCAL'
        data.setextra('{}/Gain'.format(JUPITERCAL), 
                      self.G,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('{}/Frequency'.format(JUPITERCAL), 
                      freq,
                      [Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('{}/MJD'.format(JUPITERCAL), 
                      np.array([np.nanmean(mjd)]),
                      [Types._OTHER_])
        data.setextra('{}/EL'.format(JUPITERCAL), 
                      np.array([np.nanmean(el)]),
                      [Types._OTHER_])
        data.setextra('{}/DISTANCE'.format(JUPITERCAL), 
                      np.array([dist]),
                      [Types._OTHER_])


class JupCal(AmbLoadCal):
    """
    Interpolate Ambient Measurements
    """
    def __init__(self, amb_dir='JupiterCals/', force=False):
        super().__init__()
        self.amb_dir = amb_dir
        self.force = False
        self.grp = 'JUPITERCAL'
        
    def __str__(self):
        return "Applying ambient load calibration"


    def run(self, data):
        # don't want to calibrate multiple times!
        #if ('comap' in data.data):
        #    if (not isinstance(data.data['comap'].attrs.get('cal_ambload'), type(None))) and (not self.force):
        #        print('2 {} is already ambient load calibrated'.format(data.filename.split('/')[-1]))
        #        return None

        self.Gain, self.GainFreq = self.getNearestGain(data)

        tod = data.getdset('spectrometer/tod')
        tod *= self.Gain[...,np.newaxis]


        data.setAttr('comap', 'cal_jup', True)
