import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Tools import Coordinates, Types
from os import listdir, getcwd
from os.path import isfile, join

from mpi4py import MPI 
comm = MPI.COMM_WORLD

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
            
            azval = self.interpAzEl(data.getdset('pointing/azActual'),
                                    data.getdset('pointing/MJD'),
                                    data.getdset('spectrometer/MJD'))
            elval = self.interpAzEl(data.getdset('pointing/elActual'),
                                    data.getdset('pointing/MJD'),
                                    data.getdset('spectrometer/MJD'))

            
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

        hk    = self.dout.create_group('hk')
        hkgroups = ['deTracker', 'drivenode', 'env', 'saddlebag', 'vane']
        for hkgroup in hkgroups:
            din.data.copy('hk/{}'.format(hkgroup), hk)

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

            if 'AMBIENTLOADS' in g:
                gainmjd[i] = g['AMBIENTLOADS/MJD'][0]
                gainel[i]  = g['AMBIENTLOADS/EL'][0]
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

        self.Gain = g['AMBIENTLOADS/Gain'][...]
        self.GainFreq = g['AMBIENTLOADS/Frequency'][...]
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

class AmbientLoad(DataStructure):
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

    def parsefilename(self, filename):
        """
        We want to save ambient load information separately for calibration purposes,
        therefore we must parse the filename.
        """
        prefix = filename.split('/')[-1].split('.h')[0]
        self.filename = '{}/{}{}.hdf5'.format(self.output_dir, prefix, self.suffix)

    def findHotCold(self, mtod):
        """
        Find the hot/cold sections of the ambient load scan
        """

        nSamps = mtod.size
        # Assuming there is a big step we take mid-value
        midVal = (np.nanmax(mtod)-np.nanmin(mtod))/2.

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
            raise NoHotError('No cold load data found')

        snips = int(np.min(np.where(X)[0])), int(np.max(np.where(X)[0]))
        idHot = (mtodArgsort[groupHot])[X]

        coldTod = mtod[(mtodArgsort[groupCold])]
        X =  np.abs((coldTod - np.median(coldTod))/rms) < 1
        if np.sum(X) == 0:
            raise NoColdError('No cold load data found')

        snips = int(np.min(np.where(X)[0])), int(np.max(np.where(X)[0]))
        idCold = (mtodArgsort[groupCold])[X]
        
        return idHot, idCold

    def run(self,data):

        #if not b'Tsys' in data.data['comap'].attrs['comment']:
        #   print('Not an ambient load scan')
        #    return None

        # Check if elevations shift (e.g., there is a skydip
        el  = data.getdset('pointing/elEncoder')
        e0, e1 = np.nanmin(el), np.nanmax(el)
        if (e1-e0) > self.elLim:
            print('Elevation range exceeds {:.0f} degrees'.format(self.elLim))
            return None
        

        self.parsefilename(data.filename)

        freq = data.getdset('spectrometer/frequency')
        tod = data.getdset('spectrometer/tod') # data.data['spectrometer/tod']
        mjd = data.getdset('spectrometer/MJD') # data.data['spectrometer/MJD'][:]
        self.mjd = np.mean(mjd)
        self.elevation = np.nanmedian(el)

        # Create output containers
        nHorns, nSBs, nChan, nSamps = tod.shape

        self.Tsys = np.zeros((nHorns, nSBs, nChan))
        self.G    = self.Tsys*0.

        #if nSamps < self.minSamples:
        #   return 0


        tHot  = data.getdset('hk/env/ambientLoadTemp') + self.tHotOffset
        hkMJD = data.getdset('hk/env/MJD')
        tHot  = gaussian_filter1d(tHot, 35)
        tHot  = interp1d(hkMJD, tHot, bounds_error=False, fill_value=np.nan)(mjd)

        # Fill in any NaNs that we we might encounter
        ambNan = np.where(np.isnan(tHot))[0]
        if len(ambNan) > 0:
            lowAmbNan = ambNan[ambNan < tHot.size//2]
            hiAmbNan  = ambNan[ambNan > tHot.size//2]
            if len(lowAmbNan) > 0:
                tHot[lowAmbNan] = tHot[max(lowAmbNan)+1]
            if len(hiAmbNan) > 0:
                tHot[hiAmbNan]  = tHot[min(hiAmbNan)- 1]

        tHot = np.mean(tHot)


        for i in range(nHorns):
            if (nSamps//2 - self.minSamples//2 < 0):
                continue
            itod = tod[i,:,:,:]
            
            try:
                idHot, idCold = self.findHotCold(np.nanmedian(itod[0,:,:],axis=0))
            except (NoHotError, NoColdError):
                continue

            for j in range(nSBs):
                vHot = np.nanmean(itod[j,:,idHot],axis=0)
                vCold= np.nanmean(itod[j,:,idCold],axis=0)
                Y = vHot/vCold
                #if (j==0) | (j==2):
                #    step = -1
                #else:
                #    step = 1
                self.Tsys[i,j,:] = ((tHot - self.tCold)/(Y - 1.) - self.tCold )#[::step]
                self.G[i,j,:] = ((vHot - vCold)/(tHot - self.tCold))#[::step]

        data.setextra('AMBIENTLOADS/Gain', 
                      self.G,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('AMBIENTLOADS/Tsys', 
                      self.Tsys,
                      [Types._HORNS_, 
                       Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('AMBIENTLOADS/Frequency', 
                      freq,
                      [Types._SIDEBANDS_, 
                       Types._FREQUENCY_])
        data.setextra('AMBIENTLOADS/MJD', 
                      np.array([np.nanmean(mjd)]),
                      [Types._OTHER_])
        data.setextra('AMBIENTLOADS/EL', 
                      np.array([np.nanmean(el)]),
                      [Types._OTHER_])
