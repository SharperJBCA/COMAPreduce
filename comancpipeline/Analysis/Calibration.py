import numpy as np
from matplotlib import pyplot
import h5py
import comancpipeline
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats
from os import listdir, getcwd
from os.path import isfile, join
import grp
import shutil


from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import os
from astropy.time import Time
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

from scipy.signal import find_peaks

__vane_version__ = 'v3'
__level2_version__ = 'v1'

class NoColdError(Exception):
    pass
class NoHotError(Exception):
    pass
class NoDiodeError(Exception):
    pass

def get_vane_flag(data):
    """
    Locates the calvane positions in a level 1 file

    (Extracted to be a function as this is generally useful to have)
    """
    mjd  = data['spectrometer/MJD']

    # Need to count how many calvane events occur, look for features 2**13
    if mjd[0] > Time(datetime(2019,6,14),format='datetime').mjd: # The cal vane feature bit can be trusted after 14 June 2019
        features = np.floor(np.log(data['spectrometer/features'][:])/np.log(2)).astype(int)
        justVanes = (features == 13)
    else: # Must use cal vane angle to calculate the diode positions

        if mjd[0] < Time(datetime(2019,3,1),format='datetime').mjd: # Early observations before antenna0
            hkMJD = data['hk/vane/MJD'][:]
            angles = np.interp(mjd,hkMJD, data['hk/vane/angle'][:])
        else:
            hkMJD = data['hk/antenna0/vane/utc'][:]
            angles = np.interp(mjd,hkMJD, data['hk/antenna0/vane/angle'][:])
        justVanes = (angles < 16000)

    return justVanes

class BandAverage(BaseClasses.DataStructure):
    """
    Average data in the level 2 structure in to bands. Calculate channel masks.
    """
    def __init__(self,*args,**kwargs):
        """
        """
        self.feeds_select = 'all'
        self.level2 = 'level2'

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        if self.feeds_select == 'all':
            feeds = data['level1/spectrometer/feeds'][:]
        else:
            if (not isinstance(self.feeds_select,list)) & (not isinstance(self.feeds_select,np.ndarray)) :
                self.feeds = [int(self.feeds_select)]
                feeds = self.feeds
            else:
                feeds = [int(f) for f in self.feeds_select]
        self.feeds = feeds

        vane = data[f'{self.level2}/Vane']
        print(vane.keys())
        statistics = data[f'{self.level2}/Statistics']
        frequency  = data[f'{self.level2}/frequency'][...]
        scan_edges = data[f'{self.level2}/Statistics/scan_edges'][...]
        nBands,nChans = frequency.shape
        for iscan,(start,end) in enumerate(scan_edges):
            for ifeed, feed in enumerate(self.feeds):
                if feed == 20:
                    continue
                for iband in range(nBands):

                    coeffs = statistics['filter_coefficients'][ifeed,iband,:,iscan,0]
                    fnoise = statistics['fnoise_fits'][ifeed,iband,:,iscan,:]
                    rms = statistics['wnoise_auto'][ifeed,iband,:,iscan,0]
                    tsys= self.average_vane(vane['Tsys'][0,ifeed,iband,...])
                    rms_red = rms*np.sqrt((1./fnoise[:,0])**fnoise[:,1])

                    #pyplot.plot(coeffs,rms_red,'.')
                    p=pyplot.plot(frequency[iband],rms*np.sqrt(0.05*16./1024.*1e9)-tsys)
                    #pyplot.plot(frequency[iband],tsys,color=p[0].get_color(),ls='-.')
                pyplot.show()

    def average_vane(self,tsys,nbins=64):

        return np.mean(np.reshape(tsys,(64,tsys.size//64)),axis=1)

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'


        # Need to check that there are noise stats:
        if not 'level2/Statistics' in data:
            return data

        # Want to ensure the data file is read/write
        #if not data.mode == 'r+':
        #    filename = data.filename
        #    data.close()
        #    data = h5py.File(filename,'r+')

        self.run(data)
        #self.write(data)

        return data

class CreateLevel2Cont(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1,
                 average_width=512,calvanedir='AncillaryData/CalVanes',
                 cal_mode = 'Vane', cal_prefix='',level2='level2',
                 data_dirs=None,
                 calvane_prefix='CalVane',**kwargs):
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

    def __str__(self):
        return "Creating level2 file with channel binning of {}".format(self.average_width)

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


        # Setup output file here, we will have to write in sections.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if os.path.isfile(self.outfilename):
            os.remove(self.outfilename)


        # Opening file here to write out data bit by bit
        self.i_nFeeds, self.i_nBands, self.i_nChannels,self.i_nSamples = data['spectrometer/tod'].shape
        avg_tod_shape = (self.i_nFeeds, self.i_nBands, self.i_nChannels//self.average_width, self.i_nSamples)
        self.avg_tod = np.zeros(avg_tod_shape,dtype=data['spectrometer/tod'].dtype)

        # Average the data and apply the gains
        self.average_obs(data.filename,data, self.avg_tod)

    def okay_level2_version(self,h):
        """
        Check level 2 file is up to date
        """
        
        if not 'version' in h.attrs:
            return False
        if not h.attrs['version'] == __level2_version__:
            return False
        if not 'vane-version' in h.attrs:
            return False
        if not h.attrs['vane-version'] == h['Vane'].attrs['version']:
            return False

        if not self.level2 in h:
            return False

        return True

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        self.comment = self.getComment(data)
        prefix = data.filename.split('/')[-1].split('.hd5')[0]

        data_dir_search = np.where([os.path.exists('{}/{}_Level2Cont.hd5'.format(data_dir,prefix)) for data_dir in self.data_dirs])[0]
        if len(data_dir_search) > 0:
            self.output_dir = self.data_dirs[data_dir_search[0]]
        self.outfilename = '{}/{}_Level2Cont.hd5'.format(self.output_dir,prefix)
        
        # Skip files that are already calibrated:
        if os.path.exists(self.outfilename) & (not self.overwrite): 
            self.logger(f'{fname}:{self.name}: Level 2 file exists...checking vane version...')
            self.outfile = h5py.File(self.outfilename,'r')
            if self.level2 in self.outfile:
                lvl2 = self.outfile[self.level2]
                if self.okay_level2_version(lvl2):
                    self.logger(f'{fname}:{self.name}: Vane calibration up to date. Skipping.')
                    data.close()
                    return self.outfile
                else:
                    self.logger(f'{fname}:{self.name}: Vane calibration needs updating.')
                    self.outfile.close()
            else:
                self.logger(f'{fname}:{self.name}: Level 2 file exists, vane exists but not {self.level2} group, creating...')

        self.logger(f'{fname}:{self.name}: Applying vane calibration. Bin width {self.average_width}.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing level 2 file: {self.outfilename}')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')
        # Now change to the level2 file for all future analyses.
        if data:
            data.close() # just double check we close the level 1 file.
        return self.outfile

    def average_obs(self,filename,data, tod):
        """
        Average TOD together
        """

        # --- Average down the data
        width = self.average_width
        nHorns, nSBs, nChans, nSamples = tod.shape
        nHorns = len(self.feeds)


        frequency = data['spectrometer/frequency'][...]
        self.avg_frequency = np.zeros((nSBs, nChans))
        # Averaging the data either using a Tsys/Calvane measurement or assume equal weights
        try:
            # Altered to allow Skydip files to be calibrated using the neighbouring
            # cal-vane files (following SKYDIP-LISSAJOUS obs. plan)
            if 'Sky nod' in self.comment:
                cname, gain, tsys,spikes = self.getcalibration_skydip(data)
            else:
                cname, gain, tsys,spikes = self.getcalibration_obs(data)
        except ValueError:
            cname = '{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname)
            gain = np.ones((2, nHorns, nSBs, nChans*width))
            tsys = np.ones((2, nHorns, nSBs, nChans*width))


        # Future: Astro Calibration
        # if self.cal_mode.upper() == 'ASTRO':
        # gain = self.getcalibration_astro(data)
        for ifeed, feed in enumerate(tqdm(self.feeds,desc=self.name)):
            feed_array_index = self.feed_dict[feed]
            d = data['spectrometer/tod'][feed_array_index,...] 

            for sb in range(nSBs):
                # Weights/gains already snipped to just the feeds we want
                w, g,chan_flag = 1./tsys[0,ifeed, sb, :]**2, gain[0,ifeed, sb, :], spikes[0,ifeed,sb,:]
                w[chan_flag] = 0

                gvals = np.zeros(nChans)
                for chan in range(nChans):
                    try:
                        bot = np.nansum(w[chan*width:(chan+1)*width])
                    except:
                        continue

                    if self.cal_mode.upper() == 'VANE':
                        caltod = d[sb,chan*width:(chan+1)*width,:]/g[chan*width:(chan+1)*width,np.newaxis]
                    elif self.cal_mode.upper() == 'NOCAL':
                        caltod = d[sb,chan*width:(chan+1)*width,:]
                    elif self.cal_mode.upper() == 'ASTRO':
                        pass
                    else:
                        pass

                    if width > 1:
                        self.avg_tod[ifeed,sb,chan,:] = np.sum(caltod*w[chan*width:(chan+1)*width,np.newaxis],axis=0)/bot
                    else:
                        self.avg_tod[ifeed,sb,chan,:] = caltod


                    self.avg_frequency[sb,chan] = np.mean(frequency[sb,chan*width:(chan+1)*width])

    def getcalibration_skydip(self,data):
        """
        No vane measurements in sky dips, find nearest valid obsid.
        """
        obsidSearch = int(data.filename.split('/')[-1][6:13]) + 1
        searchstring = "{:07d}".format(obsidSearch)
        calFileDir = listdir(self.calvanedir)
        calFileFil1 = [s for s in calFileDir  if searchstring in s]
        calFileName = [s for s in calFileFil1 if '.hd5' in s]

        if calFileName == []:
            return data
        cname = '{}/{}'.format(self.calvanedir,calFileName[0])
        gain_file = h5py.File(cname,'r')
        gain = gain_file['Gain'][...] # (event, horn, sideband, frequency)
        tsys = gain_file['Tsys'][...] # (event, horn, sideband, frequency) - use for weights
        spikes=gain_file['Spikes'][...]
        gain_file.close()
        return cname, gain, tsys, spikes

    def getcalibration_obs(self,data):
        """
        Open vane calibration file.
        """
        fname = data.filename.split('/')[-1]
        cname = '{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname)
        gain_file = h5py.File(cname,'r')
        gain = gain_file['Gain'][...] # (event, horn, sideband, frequency)
        tsys = gain_file['Tsys'][...] # (event, horn, sideband, frequency) - use for weights
        spikes = gain_file['Spikes'][...]
        gain_file.close()
        return cname, gain, tsys, spikes

    def okay_vane_version(self,h):
        """
        Check to see if this vane version is up to date
        """
        if not 'version' in h.attrs:
            return False
        if not 'Spikes' in h:
            return False
        if not h.attrs['version'] == __vane_version__:
            return False
        
        return True


    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """

        if os.path.exists(self.outfilename):
            self.outfile = h5py.File(self.outfilename,'a')
        else:
            self.outfile = h5py.File(self.outfilename,'w')

        # Set permissions and group
        os.chmod(self.outfilename,0o664)
        shutil.chown(self.outfilename, group='comap')


        if self.level2 in self.outfile:
            del self.outfile[self.level2]
        lvl2 = self.outfile.create_group(self.level2)

        tod_dset = lvl2.create_dataset('averaged_tod',data=self.avg_tod, dtype=self.avg_tod.dtype)
        tod_dset.attrs['Unit'] = 'K'
        tod_dset.attrs['Calibration'] = '{self.cal_mode}:{self.cal_prefix}'

        freq_dset = lvl2.create_dataset('frequency',data=self.avg_frequency, dtype=self.avg_frequency.dtype)

        # Link the Level1 data
        data_filename = data.filename
        fname = data.filename.split('/')[-1]
        data.close()
        if not 'level1' in self.outfile:
            self.outfile['level1'] = h5py.ExternalLink(data_filename,'/')
        lvl2.attrs['version'] = __level2_version__

        # Add version info
        lvl2.attrs['pipeline-version'] = comancpipeline.__version__

        # Link the Level1 data
        if 'Vane' in lvl2:
            del lvl2['Vane']
        lvl2['Vane'] = h5py.ExternalLink('{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname),'/')
        lvl2.attrs['vane-version'] = lvl2['Vane'].attrs['version']



class CalculateVaneMeasurement(BaseClasses.DataStructure):
    """
    Calculates the Tsys and Gain factors from a COMAP vane in/out measurement.
    """
    def __init__(self, output_dir='AmbientLoads/',
                 do_plots=False,
                 elLim=5, feeds = 'all',
                 minSamples=200, tCold=2.74, 
                 tHotOffset=273.15,prefix='VaneCal',**kwargs):
        super().__init__(**kwargs)

        self.name = 'CalculateVaneMeasurement'
        self.do_plots=do_plots
        self.feeds_select = feeds

        self.output_dir = output_dir
        self.prefix = prefix

        self.elLim = elLim
        self.minSamples = minSamples

        self.tCold = tCold # K, cmb
        self.tHotOffset = tHotOffset # celcius -> kelvin

    def __str__(self):
        return "Calculating Tsys and Gain from Ambient Load observation."

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')
        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)

        if os.path.exists(outfile) & (not self.overwrite):
            vane = h5py.File(outfile,'r')
            if self.okay_vane_version(vane):
                vane.close()
                self.logger(f'{fname}:{self.name}: {self.prefix}_{fname} exists, ignoring (overwrite = {self.overwrite})')
                return data
            vane.close()

        comment = self.getComment(data)

        # Ignore Sky dips
        if 'Sky nod' in comment:
            self.logger(f'{fname}:{self.name}: Observation is a sky nod (ignoring)')
            return None

        self.logger(f'{fname}:{self.name}: Measuring vane...')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing vane calibration file: {outfile}')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def okay_vane_version(self,h):
        """
        Check to see if this vane version is up to date
        """
        if not 'version' in h.attrs:
            return False
        if not 'Spikes' in h:
            return False
        if not h.attrs['version'] == __vane_version__:
            return False
        
        return True

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
        rms = stats.AutoRMS(mtod[:,None]).flatten()

        # Where is it hot? where is it cold?
        groupHot  = (mtodSort-midVal) > rms*5
        groupCold = (mtodSort-midVal) < rms*5

        # Find indices where vane is in
        hotTod = mtod[(mtodArgsort[groupHot])]
        X =  np.abs((hotTod - np.median(hotTod))/rms) < 1
        if np.sum(X) == 0:
            raise NoHotError('No hot load data found')
        snips = int(np.min(np.where(X)[0])), int(np.max(np.where(X)[0]))
        idHot = (mtodArgsort[groupHot])[X]
        idHot = np.sort(idHot)

        # Find indices where vane is out
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
        idCold = idCold[idCold > min(idHot)]
        return idHot, idCold

    def FindVane(self,data):
        """
        Find the calibration vane start and end positions.
        Raise NoDiodeError if no vane found
        """

        vane_flag = get_vane_flag(data)
        justVanes = np.where(vane_flag)[0]

        if len(justVanes) == 0:
            self.nodata = True
            raise NoDiodeError('No diode feature found')
        else:
            self.nodata = False

        nSamples2 = int((justVanes.size//2)*2)
        diffFeatures = justVanes[1:] - justVanes[:-1]
        # Anywhere where diffFeatures > 60 seconds must be another event
        timeVanes = int(60*50) # seconds * sample_rate
        events = np.where((diffFeatures > timeVanes))[0]

        nVanes = len(events) + 1

        # Calculate the cal vane start and end positions
        vanePositions = []
        for i in range(nVanes):
            if i == 0:
                low = justVanes[0]
            else:
                low = justVanes[events[i-1]+1]
            if i < nVanes-1:
                high = justVanes[events[i]]
            else:
                high = justVanes[-1]
            vanePositions += [[low,high]]

        return vanePositions, len(vanePositions)

    def run(self,data):
        """
        Calculate the cal vane calibration factors for observations after
        June 14 2019 (MJD: 58635) after which corrections to the calvane features
        were made.

        """

        fname = data.filename.split('/')[-1]

        # Read in data that is required:
        freq = data['spectrometer/frequency'][...]
        mjd  = data['spectrometer/MJD'][:]
        el   = data['pointing/elActual'][:]

        self.feeds,self.feed_inds,self.feed_dict = self.getFeeds(data,self.feeds_select)

        self.mjd = np.mean(mjd)
        self.elevation = np.nanmedian(el)

        # Keep TOD as a file object to avoid reading it all in
        tod   = data['spectrometer/tod']
        btod  = data['spectrometer/band_average']
        nHorns, nSBs, nChan, nSamps = tod.shape


        # Setup for calculating the calibration factors, interpolate temperatures:
        if mjd[0] < Time(datetime(2019,3,1),format='datetime').mjd: # Early observations before antenna0
            tHot  = np.nanmean(data['hk/env/ambientLoadTemp'][:])/100. + self.tHotOffset
            hkMJD = data['hk/env/MJD'][:]
            # Observations before vane change:
        elif (mjd[0] > Time(datetime(2019,3,1),format='datetime').mjd) & (mjd[0] < Time(datetime(2019,8,23),format='datetime').mjd):
            tHot  = np.nanmean(data['hk/antenna0/env/ambientLoadTemp'][:])/100. + self.tHotOffset
            hkMJD = data['hk/antenna0/env/utc'][:]
        else:
            tHot  = np.nanmean(data['hk/antenna0/vane/Tvane'][:])/100. + self.tHotOffset
            hkMJD = data['hk/antenna0/vane/utc'][:]

        vanePositions, nVanes = self.FindVane(data)

        self.vane_samples = np.zeros((nVanes,2))
        # Create output containers
        self.Tsys = np.zeros((nVanes, nHorns, nSBs, nChan))
        self.Gain = np.zeros((nVanes, nHorns, nSBs, nChan))
        self.RMS  = np.zeros((nVanes, nHorns, nSBs, nChan))
        self.Spikes = np.zeros((nVanes, nHorns, nSBs, nChan),dtype=bool)

        # Now loop over each event:
        for horn, feedid in enumerate(tqdm(self.feeds,desc=self.name)):
            for vane_event, (start, end) in enumerate(vanePositions):
                # Get the mean time of the event
                if horn == 0:
                    self.vane_samples[vane_event,:] = int(start), int(end)

                tod_slice = tod[horn,:,:,start:end]

                btod_slice = btod[horn,:,start:end]

                try:
                    idHot, idCold = self.findHotCold(btod_slice[0,:])
                except (NoHotError,NoColdError) as e:
                    self.logger(f'{fname}:{self.name}: No vane found in feed {feedid}.',error=e)
                    break


                time= np.arange(tod_slice.shape[-1])
                for sb in range(nSBs):
                    vHot = np.nanmedian(tod_slice[sb,:,idHot],axis=0)
                    vCold= np.nanmedian(tod_slice[sb,:,idCold],axis=0)
                    Y = vHot/vCold
                    tsys = ((tHot - self.tCold)/(Y - 1.) ) - self.tCold
                    self.Tsys[vane_event,horn,sb,:] = tsys
                    self.Gain[vane_event,horn,sb,:] = ((vHot - vCold)/(tHot - self.tCold))

                    if self.do_plots:
                        pyplot.plot(time,btod_slice[0,:],'-',color='k')
                        pyplot.plot(time[idCold],btod_slice[0,idCold],'.',color='C0',label='Cold')
                        pyplot.plot(time[idHot],btod_slice[0,idHot],'.',color='C3',label='Hot')
                        pyplot.axhline(np.nanmedian(btod_slice[0,idCold]),color='C0',lw=3,ls='--')
                        pyplot.axhline(np.nanmedian(btod_slice[0,idHot]),color='C3',lw=3,ls='--')
                        pyplot.xlabel('Sample')
                        pyplot.ylabel('RU (V)')
                        pyplot.grid()
                        pyplot.savefig('figures/Tsys_Example.png')
                        pyplot.clf()
                    
                    chans = np.arange(tsys.size)
                    peaks,properties = find_peaks(tsys,prominence=5,width=[0,60])
                    widths = (properties['right_ips']-properties['left_ips'])*2
                    bad = np.zeros(tsys.size,dtype=bool)
                    bad[[0,512,len(bad)-1]] = True
                    bad[np.arange(0,1024,64,dtype=int)] = True
                    frequency   = data['spectrometer/frequency'][sb,:]

                    if self.do_plots:
                        pyplot.plot(frequency,tsys,'k',lw=3)
                    for peak,width in zip(peaks,widths):
                        sel = np.abs(chans - peak) < width
                        bad[sel] = True
                        f0 = np.argmin((chans - (peak - width))**2)
                        f1 = np.argmin((chans - (peak + width))**2)
                        if self.do_plots:
                            pyplot.fill_between([frequency[f0],frequency[f1]],[0,0],[200,200],color='grey',alpha=0.5,hatch='/')
                    self.Spikes[vane_event,horn,sb,:] = bad

                    tod_slice[sb,:,:] /= self.Gain[vane_event,horn,sb,:,np.newaxis]

                if self.do_plots:
                    pyplot.ylim(0,200)
                    pyplot.xlabel('Frequency (GHz)')
                    pyplot.ylabel(r'$T_\mathrm{sys}$')
                    pyplot.xlim(26,30)
                    pyplot.savefig('figures/Tsys_Spikes_Example.png')
                    pyplot.show()


    def write(self,data):
        """
        Write the Tsys, Gain and RMS to a pandas data frame for each hdf5 file.
        """
        if self.nodata:
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # We will store these in a separate file and link them to the level2s
        fname = data.filename.split('/')[-1]
        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)
        if os.path.exists(outfile):
            os.remove(outfile)

        output = h5py.File(outfile,'a')

        # Set permissions and group
        os.chmod(outfile,0o664)
        shutil.chown(outfile, group='comap')

        # Store datasets in root
        dnames = ['Tsys','Gain','VaneEdges','Spikes']
        dsets  = [self.Tsys, self.Gain, self.vane_samples,self.Spikes]
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)

        output['Tsys'].attrs['Unit'] = 'K'
        output['Gain'].attrs['Unit'] = 'V/K'
        output['VaneEdges'].attrs['Unit'] = 'Index'
        output['Spikes'].attrs['Unit'] = 'bool'
        output.attrs['version'] = __vane_version__
                        
        output.close()



class CreateLevel2RRL(CreateLevel2Cont):
    """
    Takes level 1 files and select RRL channels, filters data around lines.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1,
                 average_width=512,calvanedir='AncillaryData/CalVanes',
                 cal_mode = 'Vane', cal_prefix='',level2='level2rrl',
                 calvane_prefix='CalVane',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)

        self.name = 'CreateLevel2RRL'
        self.feeds_select = feeds

        self.output_dir = output_dir

        self.calvanedir = calvanedir
        self.calvane_prefix = calvane_prefix

        self.cal_mode = cal_mode
        self.cal_prefix=cal_prefix

        self.level2=level2

        R = 10973731.6
        c = 3e8 
        #self.rrl_qnumbers = np.array([58,59,60,61,62])
        #self.rrl_frequencies = R*c*(1./self.rrl_qnumbers**2 - 1./(self.rrl_qnumbers+1)**2)/1e9
        self.rrl_frequencies=np.array([26.93916209,	
                                       28.27486961,	
                                       29.70036235,	
                                       31.22331322,	
                                       32.85219558])
        self.rrl_qnumbers=np.array([62,61,60,59,58])
        self.binwidth = 3 # store three bins for each line
        

    def __str__(self):
        return "Creating level2 RRL file"

    def frequency2velocity(self,rrl_freq,freqs):
        cspeed = 299792458.0
        return (rrl_freq-freqs)/rrl_freq*cspeed/1e3

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


        # Setup output file here, we will have to write in sections.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if os.path.isfile(self.outfilename):
            os.remove(self.outfilename)


        # Opening file here to write out data bit by bit
        self.i_nFeeds, self.i_nBands, self.i_nChannels,self.i_nSamples = data['spectrometer/tod'].shape

        # Average the data and apply the gains
        self.average_obs(data.filename,data)

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting.')

        self.comment = self.getComment(data)
        prefix = data.filename.split('/')[-1].split('.hd5')[0]
        self.outfilename = '{}/{}_Level2RRL.hd5'.format(self.output_dir,prefix)

        
        # Skip files that are already calibrated:
        if os.path.exists(self.outfilename) & (not self.overwrite): 
            self.logger(f'{fname}:{self.name}: Level 2 file exists...checking vane version...')
            self.outfile = h5py.File(self.outfilename,'r')
            lvl2 = self.outfile[self.level2]
            if self.okay_level2_version(lvl2):
                self.logger(f'{fname}:{self.name}: Vane calibration up to date. Skipping.')
                data.close()
                return self.outfile
            else:
                self.outfile.close()

        self.logger(f'{fname}:{self.name}: Applying vane calibration.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing level 2 file: {self.outfilename}')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')
        # Now change to the level2 file for all future analyses.
        if data:
            data.close() # just double check we close the level 1 file.
        return #self.outfile

    def average_obs(self,filename,data):
        """
        Average TOD together
        """

        # --- Average down the data
        nHorns, nSBs, nChans, nSamples = data['spectrometer/tod'].shape
        nHorns = len(self.feeds)


        frequencies = data['spectrometer/frequency'][...]
        # Averaging the data either using a Tsys/Calvane measurement or assume equal weights
        try:
            # Altered to allow Skydip files to be calibrated using the neighbouring
            # cal-vane files (following SKYDIP-LISSAJOUS obs. plan)
            if 'Sky nod' in self.comment:
                cname, gain, tsys,spikes = self.getcalibration_skydip(data)
            else:
                cname, gain, tsys,spikes = self.getcalibration_obs(data)
        except ValueError:
            cname = '{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname)
            gain = np.ones((2, nHorns, nSBs, nChans))
            tsys = np.ones((2, nHorns, nSBs, nChans))

        crval = [(18 + 47./60.+34.81/60.**2)*15,-(1 + 56./60.+31./60.**2)]

        # Future: Astro Calibration
        # if self.cal_mode.upper() == 'ASTRO':
        # gain = self.getcalibration_astro(data)
        self.output = np.zeros((nHorns,len(self.rrl_frequencies),11,nSamples))
        self.spectra  = np.zeros((len(self.feeds),len(self.rrl_qnumbers),21,nSamples))*np.nan
        self.velocity = np.zeros((len(self.feeds),len(self.rrl_qnumbers),21))*np.nan
        for ifeed, feed in enumerate(tqdm(self.feeds,desc=self.name)):
            feed_array_index = self.feed_dict[feed]
            ra  = data['spectrometer/pixel_pointing/pixel_ra'][feed_array_index,...]
            dec = data['spectrometer/pixel_pointing/pixel_dec'][feed_array_index,...] 

            select = np.where((Coordinates.AngularSeperation(ra,dec,crval[0],crval[1]) < 1.5/60.))[0]
            d = data['spectrometer/tod'][feed_array_index,...] 
            #for rllfreq,frequency in enumerate(self.rrl_frequencies[1:2]):
            for iqno,(qno,rrl_freq) in enumerate(zip(self.rrl_qnumbers,self.rrl_frequencies)):
                for sb in range(nSBs):
                    # Weights/gains already snipped to just the feeds we want
                    if (rrl_freq > np.min(frequencies[sb])) & (rrl_freq < np.max(frequencies[sb])):
                        w, g,chan_flag = 1./tsys[0,ifeed, sb, :]**2, gain[0,ifeed, sb, :], spikes[0,ifeed,sb,:]
                        w[chan_flag] = 0
                        z = d[sb,:,:]
                        s1 = z[:,:]/g[:,None]
                        velocity = self.frequency2velocity(rrl_freq,frequencies[sb,:])
                        ichan = np.argmin((rrl_freq - frequencies[sb,:])**2)
                        lo = int(np.max([ichan - 10,0]))
                        hi = int(np.min([ichan + 11,nChans]))
                    
                        #print(ifeed,sb,qno,lo,hi)
                        self.spectra[ifeed,iqno,:(hi-lo),:] = s1[lo:hi]
                        self.velocity[ifeed,iqno,:(hi-lo)]  = velocity[lo:hi]

        #np.save('figures/RRLs/{}.npy'.format(data.filename.split('/')[-1].split('.h')[0]),{'spectra':spectra,'frequencies':frequencies,'rrls':self.rrl_frequencies})
        #np.savez('figures/RRLs/FullData_{}.npy'.format(data.filename.split('/')[-1].split('.h')[0]),spectra=self.spectra,velocity=self.velocity,rrls=self.rrl_frequencies)
        #stop
        #pyplot.plot(frequencies[sb,:],spectra.T)
        #pyplot.plot(frequencies[sb,:],np.nanmean(spectra[:19],axis=0),'k',lw=3)
        #pyplot.axvline(frequency,color='r')
        #pyplot.show()
    def write(self,data):
        """
        Write out the averaged TOD to a Level2 RRL file with an external link to the original level 1 data
        """
        if os.path.exists(self.outfilename):
            self.outfile = h5py.File(self.outfilename,'a')
        else:
            self.outfile = h5py.File(self.outfilename,'w')

        # Set permissions and group
        os.chmod(self.outfilename,0o664)
        shutil.chown(self.outfilename, group='comap')

        print(self.outfilename)
        if self.level2 in self.outfile:
            del self.outfile[self.level2]
        print('Writing...')
        lvl2 = self.outfile.create_group(self.level2)

        tod_dset = lvl2.create_dataset('tod',data=self.spectra, dtype=self.spectra.dtype)
        tod_dset.attrs['Unit'] = 'K'
        tod_dset.attrs['Calibration'] = '{self.cal_mode}:{self.cal_prefix}'
        tod_dset = lvl2.create_dataset('velocity',data=self.velocity, dtype=self.velocity.dtype)
        tod_dset.attrs['Unit'] = 'km/s'

        freq_dset = lvl2.create_dataset('frequency',data=self.rrl_frequencies, dtype=self.rrl_frequencies.dtype)

        # Link the Level1 data
        data_filename = data.filename
        fname = data.filename.split('/')[-1]
        data.close()
        if not 'level1' in self.outfile:
            self.outfile['level1'] = h5py.ExternalLink(data_filename,'/')
        lvl2.attrs['version'] = __level2_version__

        # Add version info
        lvl2.attrs['pipeline-version'] = comancpipeline.__version__

        # Link the Level1 data
        if 'Vane' in lvl2:
            del lvl2['Vane']
        lvl2['Vane'] = h5py.ExternalLink('{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname),'/')
        lvl2.attrs['vane-version'] = lvl2['Vane'].attrs['version']

