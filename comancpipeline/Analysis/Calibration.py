import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Tools import Coordinates
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
        print(missingFields)
        if not isinstance(missingFields, type(None)):
            spec = data.data['spectrometer']
            nHorn, nSB, nChan, nSample = spec['tod'].shape
            
            azval = self.interpAzEl(data.data['pointing/azActual'][...],
                                    data.data['pointing/MJD'][...],
                                    data.data['spectrometer/MJD'][...])
            elval = self.interpAzEl(data.data['pointing/elActual'][...],
                                    data.data['pointing/MJD'][...],
                                    data.data['spectrometer/MJD'][...])

            
            #az,el, ra, dec = [np.zeros(( nHorn, nSample)) for i in range(4)]
            for field in missingFields:
                if field in data.data:
                    del data.data[field]
                data.data.create_dataset(field, (nHorn, nSample), dtype='f')

            for i in range(nHorn):
                data.data[missingFields[1]][i,:] = elval+self.focalPlane.offsets[i][1] - self.focalPlane.eloff
                data.data[missingFields[0]][i,:] = azval+self.focalPlane.offsets[i][0]/np.cos(data.data[missingFields[1]][i,:]*np.pi/180.) - self.focalPlane.azoff
                    
                # fill in the ra/dec fields
                data.data[missingFields[2]][i,:],data.data[missingFields[3]][i,:] = Coordinates.h2e(data.data[missingFields[0]][i,:],
                                                                                                    data.data[missingFields[1]][i,:],
                                                                                                    data.data['spectrometer/MJD'][...], 
                                                                                                    self.longitude, 
                                                                                                    self.latitude)


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
        justFilename = data.filename.split('/')[-1]
        # try:
        #     # First see if there is a downsampled file that already exists with
        #     # a downsampled tag. If so we should do nothing.
        #     #dout = h5py.File('{}/{}'.format(self.out_dir, justFilename),'r')
        #     #targFileOkay = isinstance(dout['comap'].attrs.get('downsampled'), type(None))
        #     #dout.close()
        #     print('File {} already exists'.format(justFilename))
        #     return None
        # except OSError:
        #     targFileOkay = True

        # thisFileOkay = isinstance(data.data['comap'].attrs.get('downsampled'), type(None))
        # if (not thisFileOkay) or (not targFileOkay):
        #     print('File {} already downsampled'.format(justFilename))
        #     # Assume that if you asked for down sampled data then you want to 
        #     # actually being working with the previously downsampled dataset.
        #     data.update('{}/{}'.format(self.out_dir, justFilename))
        #     return None

        # Open the output filename
        self.dout = h5py.File('{}/{}'.format(self.out_dir, justFilename))
        self.createDatasets(data)

        # down sample the frequency based values
        spec    = data.data['spectrometer']
        outSpec = self.dout['spectrometer']
        
        freq = spec['frequency'][...]
        freq = freq.reshape((freq.shape[0], freq.shape[1]//self.factor, self.factor))
        outSpec.create_dataset('frequency', data=np.mean(freq, axis=-1))

        time_average = spec['time_average'][...]
        time_average = time_average.reshape((time_average.shape[0], time_average.shape[1],
                              time_average.shape[2]//self.factor, 
                              self.factor))
        outSpec.create_dataset('time_average', data=np.nanmean(time_average, axis=-1))
        
        # can't read in all of the TOD usually...
        tod = spec['tod']
        todOut = outSpec.create_dataset('tod', (tod.shape[0], tod.shape[1], tod.shape[2]//self.factor, tod.shape[3]))

        # ...read in one horn at a time
        nHorns, nSBs, nChans, nSamps = tod.shape
        import time
        for i in range(nHorns):
            # need frequency to be the last index...
            for j in range(nSBs):
                start=time.time()
                temp = np.transpose(tod[i,j,:,:], (1,0))

                temp = np.reshape(temp, (temp.shape[0],
                                         temp.shape[1]//self.factor, 
                                         self.factor))
                todOut[i,j,:,:] = np.transpose(np.nanmean(temp,axis=-1),(1,0))
                print(i,j,'Time taken {}'.format(time.time()-start))

        # add some new attributes:
        if data.rank == 0:
            self.dout['comap'].attrs.create('downsampled', True)
            self.dout['comap'].attrs.create('ds_factor', self.factor)
        print(self.dout.filename)
        self.dout.close()

        # At the end of downsampling, change the input data reference
        # to point to the downsampled data reference:
        #data.update('{}/{}'.format(self.out_dir, justFilename))

class AmbLoadCal(DataStructure):
    """
    Interpolate Ambient Measurements
    """
    def __init__(self, amb_dir='AmbientLoads/', force=False):
        super().__init__()
        self.amb_dir = amb_dir
        self.force = False
        
        self.fields = {'spectrometer/tod':True, 
                       'pointing/MJD':False}

    def __str__(self):
        return "Applying ambient load calibration"

    def getNearestGain(self, data):
        meanmjd = np.nanmean(data.dsets['pointing/MJD'])

        # First get the mean MJD of the data
        obsids = np.array([int(f.split('-')[-5]) for f in listdir(self.amb_dir) if isfile(join(self.amb_dir, f)) if 'hdf5' in f ])
        obsid  = int(data.filename.split('-')[-5])

        gainfiles = [f for f in listdir(self.amb_dir) if isfile(join(self.amb_dir, f)) if 'hdf5' in f]

        # find the 10 closest obsids:
        argmins = np.argsort((obsids - obsid)**2)[:10]
        gainmjd = np.zeros(len(argmins))
        gainel = np.zeros(len(argmins))
        for i,j in enumerate(argmins):
            g = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[j]),'r')
            gainmjd[i] = g['MJD'][0]
            gainel[i]  = g['EL'][0]
            g.close()

        Amass = 1./np.sin(gainel*np.pi/180.)

        AmassRange = np.where((Amass < 1.005))[0] # almost zenith

        # Now find the nearest mjd to obs
        minmjd = np.argmin((gainmjd[AmassRange] - meanmjd)**2)
        gain = h5py.File('{}/{}'.format(self.amb_dir, gainfiles[AmassRange[minmjd]]),'r')
        self.Gain = gain['Gain'][...]
        gain.close()
        
        return self.Gain

    def run(self, data):
        # don't want to calibrate multiple times!
        if ('comap' in data.data):
            if (not isinstance(data.data['comap'].attrs.get('cal_ambload'), type(None))) and (not self.force):
                print('2 {} is already ambient load calibrated'.format(data.filename.split('/')[-1]))
                return None

        self.Gain = self.getNearestGain(data)

        # if data is downsampled then we must down sample the gain too
        if not isinstance(data.data['comap'].attrs.get('downsampled'), type(None)):
            self.factor = int(data.data['comap'].attrs.get('ds_factor'))
            self.Gain = np.mean(np.reshape(self.Gain, (self.Gain.shape[0],
                                                       self.Gain.shape[1], 
                                                       self.Gain.shape[2]//self.factor, 
                                                       self.factor)),axis=-1)
        
        nHorns, nSBs, nChans, nSamples = data.data['spectrometer/tod'].shape
        field = 'spectrometer/tod'
        selectAxes, splitAxis = data.selectAxes[field], data.splitAxis[field]
        if not isinstance(selectAxes, type(None)):
            for i in selectAxes:
                self.Gain = self.Gain[i]


        data.dsets['spectrometer/tod'] /= np.take(self.Gain,
                                                  range(data.lo[field],data.hi[field]),
                                                  axis=data.splitAxis[field])[...,np.newaxis]

        data.setOutputAttr('comap', 'cal_ambload', True)


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
        self.minSamples = 2000

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

        if not b'Tsys' in data.data['comap'].attrs['comment']:
            print('Not an ambient load scan')
            return None

        # Check if elevations shift (e.g., there is a skydip
        e0, e1 = np.nanmin(data.data['pointing/elEncoder']), np.nanmax(data.data['pointing/elEncoder'])
        if (e1-e0) > self.elLim:
            print('Elevation range exceeds {:.0f} degrees'.format(self.elLim))
            return None
        

        self.parsefilename(data.filename)

        tod = data.data['spectrometer/tod']
        mjd = data.data['spectrometer/MJD'][:]
        self.mjd = np.mean(mjd)
        self.elevation = np.nanmedian(data.data['pointing/elEncoder'])

        # Create output containers
        nHorns, nSBs, nChan, nSamps = tod.shape

        self.Tsys = np.zeros((nHorns, nSBs, nChan))
        self.G    = self.Tsys*0.

        if nSamps < self.minSamples:
            return 0


        tHot = data.data['hk']['env']['ambientLoadTemp'][:] + self.tHotOffset
        hkMJD = data.data['hk']['env']['MJD'][:]
        tHot = gaussian_filter1d(tHot, 35)
        tHot = interp1d(hkMJD, tHot, bounds_error=False, fill_value=np.nan)(mjd)

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
            if (nSamps//2 - 1000 < 0):
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

        #write to disk
        self.writeFile(self.filename, self.overwrite)
        data.stop = True # if this was a calibrator, we dont want to run anything else

    def writeFile(self, filename, overwrite=True):
        dout = h5py.File(filename,'a')
        if overwrite & (len(dout.keys()) > 0):
            for k in dout.keys():
                del dout[k]
        dout['Tsys'] = self.Tsys
        dout['Gain'] = self.G
        dout['MJD']  = np.array([self.mjd])
        dout['EL']   = np.array([self.elevation])
        dout.close()
