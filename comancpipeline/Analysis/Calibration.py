import numpy as np
from matplotlib import pyplot
import h5py
import comancpipeline
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting
from comancpipeline.Tools import Coordinates, Types, stats, FileTools
from os import listdir, getcwd
from os.path import isfile, join
import grp
import shutil
import os

from scipy.interpolate import interp1d
import datetime
from tqdm import tqdm
import os
from astropy.time import Time
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

from scipy.signal import find_peaks

__vane_version__ = 'v4'
__level2_version__ = 'v3'

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


class CreateLevel2Cont(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all', 
                 output_obsid_starts = [0],
                 output_obsid_ends   = [None],
                 output_dirs = ['.'],
                 nworkers= 1,
                 database=None,
                 average_width=512,
                 calvanedir='CalVanes',
                 cal_mode = 'Vane', 
                 cal_prefix='',level2='level2',
                 data_dirs=None,
                 set_permissions=True,
                 permissions_group='comap',
                 calvane_prefix='CalVane',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)

        self.name = 'CreateLevel2Cont'
        self.feeds_select = feeds

        # We will be writing level 2 data to multiple drives,
        #  drive to write to will be set by the obsid of the data
        self.output_dirs = output_dirs
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends

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

        self.database   = database + '_{}'.format(os.get_pid())

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

        # Opening file here to write out data bit by bit
        self.i_nFeeds, self.i_nBands, self.i_nChannels,self.i_nSamples = data['spectrometer/tod'].shape
        avg_tod_shape = (self.i_nFeeds, self.i_nBands, self.i_nChannels//self.average_width, self.i_nSamples)
        self.avg_tod = np.zeros(avg_tod_shape,dtype=data['spectrometer/tod'].dtype)
        self.avg_rms = np.zeros((avg_tod_shape[0],avg_tod_shape[1],avg_tod_shape[2]),dtype=data['spectrometer/tod'].dtype)

        # Average the data and apply the gains
        self.average_obs(data.filename,data, self.avg_tod)

    def okay_level2_version(self,h):
        """
        Check level 2 file is up to date
        """
        if not self.level2 in h:
            return False

        if not 'version' in h[self.level2].attrs:
            return False
        if not h[self.level2].attrs['version'] == __level2_version__:
            return False
        if not 'vane-version' in h[self.level2].attrs:
            return False
        if not h[self.level2].attrs['vane-version'] == h[self.level2]['Vane'].attrs['version']:
            return False

        return True

    def _legacyJun2022_check_level2(self,outfilename):
        """
        Eventually this function can be removed.

        Checks to see if the avg_frequency dataset is in the level 2 file,
        if so it updates the level 2 version to the latest value. If not,
        the file will be updated to the latest version. 
        """
        if os.path.exists(outfilename):
            h = h5py.File(outfilename,'a')
            if 'averaged_frequency' in h[self.level2]:
                print('Found: updating level2')
                h[self.level2].attrs['version'] = __level2_version__
            else:
                print('NO AVERVAGED FREQUECNCY, WILL UPDATE')
            h.close()

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')


        self.obsid = self.getObsID(data)
        self.comment = self.getComment(data)
        prefix = data.filename.split('/')[-1].split('.hd5')[0]

        self.output_dir = self.getOutputDir(self.obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)
        self.calvanepath= '{}/{}'.format(self.output_dir,self.calvanedir)

        data_dir_search = np.where([os.path.exists('{}/{}_Level2Cont.hd5'.format(data_dir,prefix)) for data_dir in self.data_dirs])[0]
        if len(data_dir_search) > 0:
            self.output_dir = self.data_dirs[data_dir_search[0]]
        self.outfilename = '{}/{}_Level2Cont.hd5'.format(self.output_dir,prefix)        

        # Legacy function: Check if the level2 file has avg_frequency, if so continue, 
        #  else update the level2 data
        self._legacyJun2022_check_level2(self.outfilename)


        # Skip files that are already calibrated:        
        if os.path.exists(self.outfilename) & (not self.overwrite): 
            self.logger(f'{fname}:{self.name}: Level 2 file exists...checking vane version...')
            self.outfile = h5py.File(self.outfilename,'r')
            if self.level2 in self.outfile:
                if self.okay_level2_version(self.outfile):
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

        if not isinstance(self.database,type(None)):
            self.write_database(self.outfile,self.database) # pass it the new level 2 file

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
        #if True:
            # Altered to allow Skydip files to be calibrated using the neighbouring
            # cal-vane files (following SKYDIP-LISSAJOUS obs. plan)
            if 'Sky nod' in self.comment:
                print('hello sky nod')
                cname, gain, tsys,rms,spikes = self.getcalibration_skydip(data)
            else:
                cname, gain, tsys,rms,spikes = self.getcalibration_obs(data)
        except ValueError:
           cname = '{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,fname)
           gain = np.ones((2, nHorns, nSBs, nChans*width))
           tsys = np.ones((2, nHorns, nSBs, nChans*width))
           spikes = np.zeros((2, nHorns, nSBs, nChans*width),dtype=bool)
           rms = np.ones((2, nHorns, nSBs, nChans*width))

        for ifeed, feed in enumerate(tqdm(self.feeds,desc=self.name)):
            feed_array_index = self.feed_dict[feed]
            d = data['spectrometer/tod'][feed_array_index,...] 

            for sb in range(nSBs):
                # Weights/gains already snipped to just the feeds we want
                w, g,chan_flag = 1./rms[0,ifeed, sb, :]**2, gain[0,ifeed, sb, :], spikes[0,ifeed,sb,:]
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
                        self.avg_rms[ifeed,sb,chan] = np.sqrt(1./bot)
                    else:
                        self.avg_tod[ifeed,sb,chan,:] = caltod
                        self.avg_rms[ifeed,sb,chan] = np.sqrt(1./w)

                    self.avg_frequency[sb,chan] = np.mean(frequency[sb,chan*width:(chan+1)*width])

    def getcalibration_skydip(self,data):
        """
        No vane measurements in sky dips, find nearest valid obsid.
        """
        obsidSearch = int(data.filename.split('/')[-1][6:13]) + 1
        searchstring = "{:07d}".format(obsidSearch)
        calFileDir = listdir(self.calvanepath)
        calFileFil1 = [s for s in calFileDir  if searchstring in s]
        calFileName = [s for s in calFileFil1 if '.hd5' in s]

        if calFileName == []:
            return data
        cname = '{}/{}'.format(self.calvanepath,calFileName[0])
        gain_file = h5py.File(cname,'r')
        gain = gain_file['Gain'][...] # (event, horn, sideband, frequency)
        tsys = gain_file['Tsys'][...] # (event, horn, sideband, frequency) - use for weights
        spikes=gain_file['Spikes'][...]
        gain_file.close()

        tsamp = float(data['comap'].attrs['tsamp'])
        bw = 2e9/1024
        rms = tsys/np.sqrt(tsamp*bw)

        return cname, gain, tsys, rms, spikes

    def getcalibration_obs(self,data):
        """
        Open vane calibration file.
        """
        fname = data.filename.split('/')[-1]
        cname = '{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,fname)
        gain_file = h5py.File(cname,'r')
        gain = gain_file['Gain'][...] # (event, horn, sideband, frequency)
        tsys = gain_file['Tsys'][...] # (event, horn, sideband, frequency) - use for weights
        spikes = gain_file['Spikes'][...]
        gain_file.close()

                
        tsamp = float(data['comap'].attrs['tsamp'])
        bw = 2e9/1024

        rms = tsys/np.sqrt(tsamp*bw)

        return cname, gain, tsys, rms, spikes

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

    @staticmethod
    def write_database(data,database):
        """
        Write the fits to the general database
        """

        if not os.path.exists(database):
            output = FileTools.safe_hdf5_open(database,'w')
        else:
            output = FileTools.safe_hdf5_open(database,'a')

        obsid = BaseClasses.DataStructure.getObsID(data)
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)

        if 'level2' in grp:
            lvl2 = grp['level2']
        else:
            lvl2 = grp.create_group('level2')
        
        grp.attrs['level1_filename'] = data['level1'].file.filename
        grp.attrs['level2_filename'] = data.filename

        for dname, dset in data['level1/comap'].attrs.items():
            if dname in lvl2:
                del lvl2.attrs[dname]
            lvl2.attrs[dname] = dset
        output.close()

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """
        if os.path.exists(self.outfilename):
            try:
                self.outfile = h5py.File(self.outfilename,'a')
            except OSError: # sometimes files can be come corrupted, in which case start fresh
                os.remove(self.outfilename)
                self.outfile = h5py.File(self.outfilename,'w')
        else:
            self.outfile = h5py.File(self.outfilename,'w')

        # Set permissions and group
        if self.set_permissions:
            try:
                os.chmod(self.outfilename,0o664)
                shutil.chown(self.outfilename, group=self.permissions_group)
            except PermissionError:
                self.logger(f'{self.name}: Warning, couldnt set the file permissions.')


        if self.level2 in self.outfile:
            lvl2 = self.outfile[self.level2]
        else:
            lvl2 = self.outfile.create_group(self.level2)

        dsets = {'averaged_tod': self.avg_tod,
                 'averaged_rms': self.avg_rms,
                 'averaged_frequency': self.avg_frequency}
        for dname,dset in dsets.items():
            if dname in lvl2:
                del lvl2[dname]
            lvl2.create_dataset(dname,data=dset)
        lvl2.attrs['Calibration'] = '{self.cal_mode}:{self.cal_prefix}'

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
        lvl2['Vane'] = h5py.ExternalLink('{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,fname),'/')
        lvl2.attrs['vane-version'] = lvl2['Vane'].attrs['version']



class CalculateVaneMeasurement(BaseClasses.DataStructure):
    """
    Calculates the Tsys and Gain factors from a COMAP vane in/out measurement.
    """
    def __init__(self,do_plots=False,
                 calvanedir = 'CalVanes',
                 output_obsid_starts = [0],
                 output_obsid_ends   = [None],
                 output_dirs = ['.'],
                 database   = None,
                 elLim=5, feeds = 'all',
                 minSamples=200, 
                 tCold=2.73, 
                 set_permissions=True,
                 permissions_group='comap',
                 tHotOffset=273.15,
                 vaneprefix='VaneCal',**kwargs):
        super().__init__(**kwargs)

        self.name = 'CalculateVaneMeasurement'
        self.do_plots=do_plots
        self.feeds_select = feeds

        self.vaneprefix = vaneprefix

        self.elLim = elLim
        self.minSamples = minSamples

        self.tCold = tCold # K, cmb
        self.tHotOffset = tHotOffset # celcius -> kelvin

        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

        self.database = database + '_{}'.format(os.get_pid())
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends
        self.output_dirs = output_dirs
        self.calvanedir  = calvanedir

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
        self.obsid = self.getObsID(data)

        self.output_dir = self.getOutputDir(self.obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)
        self.output_dir = '{}/{}'.format(self.output_dir, self.calvanedir)
        outfile = '{}/{}_{}'.format(self.output_dir,self.vaneprefix,fname)

        if os.path.exists(outfile) & (not self.overwrite):
            vane = h5py.File(outfile,'r')
            if self.okay_vane_version(vane):
                vane.close()
                self.logger(f'{fname}:{self.name}: {self.vaneprefix}_{fname} exists, ignoring (overwrite = {self.overwrite})')
                return data
            vane.close()

        self.obsid = self.getObsID(data)
        comment    = self.getComment(data)

        self.output_dir = self.getOutputDir(self.obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)
        self.output_dir = '{}/{}'.format(self.output_dir, self.calvanedir)
        # Ignore Sky dips
        if 'Sky nod' in comment:
            self.logger(f'{fname}:{self.name}: Observation is a sky nod (ignoring)')
            return None

        self.logger(f'{fname}:{self.name}: Measuring vane...')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing vane calibration file: {outfile}')
        self.write(data)
        if not isinstance(self.database,type(None)):
            dnames = ['Tsys','Gain','VaneEdges','Spikes','Elevation']
            dsets  = [self.Tsys, self.Gain, self.vane_samples,self.Spikes,self.elevations]
            dataout = {k:v for (k,v) in zip(dnames,dsets)}
            self.write_database(data,self.database,dataout)
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
        el    = data['spectrometer/pixel_pointing/pixel_el'][0,:]
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

        self.elevations = np.zeros((nVanes))
        for vane_event, (start,end) in enumerate(vanePositions):
            self.elevations[vane_event] = np.nanmean(el[start:end])

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
                        pass

                    chans = np.arange(tsys.size)
                    peaks,properties = find_peaks(tsys,prominence=5,width=[0,60])
                    widths = (properties['right_ips']-properties['left_ips'])*2
                    bad = np.zeros(tsys.size,dtype=bool)
                    bad[[0,512,len(bad)-1]] = True
                    bad[np.arange(0,1024,64,dtype=int)] = True
                    frequency   = data['spectrometer/frequency'][sb,:]

                    if self.do_plots:
                        pass

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
        outfile = '{}/{}_{}'.format(self.output_dir,self.vaneprefix,fname)
        if os.path.exists(outfile):
            output = h5py.File(outfile,'a')
        else:
            output = h5py.File(outfile,'w')


        # Set permissions and group
        if self.set_permissions:
            try:
                os.chmod(outfile,0o664)
                shutil.chown(outfile, group=self.permissions_group)
            except PermissionError:
                self.logger(f'{self.name}: Warning, couldnt set the file permissions.')

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

    @staticmethod
    def write_database(data,database,outputdata):
        """
        Write the fits to the general database
        """

        if not os.path.exists(database):
            output = FileTools.safe_hdf5_open(database,'w')
        else:
            output = FileTools.safe_hdf5_open(database,'a')

        obsid = BaseClasses.DataStructure.getObsID(data)
        if obsid in output:
            grp = output[obsid]
        else:
            grp = output.create_group(obsid)

        if 'Vane' in grp:
            vane = grp['Vane']
        else:
            vane = grp.create_group('Vane')

        for dname, dset in  outputdata.items(): #zip(dnames, dsets):
            if dname in vane:
                del vane[dname]
            vane.create_dataset(dname,  data=dset)

        output.close()


class CreateLevel2RRL(CreateLevel2Cont):
    """
    Takes level 1 files and select RRL channels, filters data around lines.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1,
                 database   = None,
                 average_width=512,
                 calvanedir='AncillaryData/CalVanes',
                 cal_mode = 'Vane', 
                 cal_prefix='',
                 level2='level2rrl',
                 cal_source='taua',
                 data_dirs = None,
                 set_permissions=True,
                 permissions_group='comap',
                 calvane_prefix='CalVane',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)

        self.name = 'CreateLevel2RRL'
        self.feeds_select = feeds

        self.output_dir = output_dir
        if isinstance(data_dirs,type(None)):
            self.data_dirs = [self.output_dir]
        else:
            if isinstance(data_dirs,list):
                self.data_dirs = data_dirs
            else:
                self.data_dirs = [data_dirs]

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
        self.cal_source = cal_source

        self.set_permissions = set_permissions
        self.permissions_group=permissions_group

    def __str__(self):
        return "Creating level2 RRL file"

    def frequency2velocity(self,rrl_freq,freqs):
        cspeed = 299792458.0
        return (rrl_freq-freqs)/rrl_freq*cspeed/1e3

    def calibrate_data(self,data):
        """
        Calibrates data using a given calibration source
        """

        this_obsid = int(self.getObsID(data))
        # Get Gain Calibration Factors
        nfeeds = len(self.feeds)
        self.cal_factors = np.zeros((nfeeds,len(self.rrl_qnumbers),21))

        for iqno in range(len(self.rrl_qnumbers)):
            for ifeed,feed_num in enumerate(self.feeds):
                obsids = Data.feed_gains[self.cal_source.lower()]['obsids']*1
                gains  = Data.feed_gains[self.cal_source.lower()]['gains'][:,feed_num-1,:]
                frequency = np.arange(8)+0.5 + 26. #Data.feed_gains[self.cal_source.lower()]['frequency'][...]
                # now find the nearest non-nan obsid to calibrate off
                obs_idx = np.argmin((obsids - this_obsid)**2)
                
                x,y = frequency.flatten(), gains[obs_idx].flatten()
                xsort = np.argsort(x)
                x,y = x[xsort],y[xsort]
                gd = np.isfinite(x) & np.isfinite(y)
                try:
                    mdl = interp1d(x[gd],y[gd],kind='nearest', bounds_error=False, fill_value='extrapolate')
                except ValueError:
                    continue
                gain_interp = mdl(self.frequency[ifeed,iqno,:])
                self.cal_factors[ifeed,iqno,...] = gain_interp
        self.spectra = self.spectra/self.cal_factors[:,:,:,None]


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
        fname = data.filename.split('/')[-1]

        self.logger(f'{fname}:{self.name}: About to get feeds')
        self.feeds, self.feed_index, self.feed_dict = self.getFeeds(data,self.feeds_select)


        # Setup output file here, we will have to write in sections.
        self.logger(f'{fname}:{self.name}: About to create {self.output_dir}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Opening file here to write out data bit by bit
        self.logger(f'{fname}:{self.name}: About to get tod shape')
        self.i_nFeeds, self.i_nBands, self.i_nChannels,self.i_nSamples = data['spectrometer/tod'].shape

        # Average the data and apply the gains
        self.logger(f'{fname}:{self.name}: About to run average_obs')
        self.average_obs(data.filename,data)
        self.logger(f'{fname}:{self.name}: average_obs done.')
        self.logger(f'{fname}:{self.name}: About to run calibrate_data')
        self.calibrate_data(data)
        self.logger(f'{fname}:{self.name}: calibrate_data done.')

    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        # We need to be sure we are being passed a LEVEL 1 object
        if not 'level1' in data: 
            self.logger(f"{fname}:{self.name}: Not a level 2 file, perhaps you've not run CreateLevel2Cont?")
            return data

        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting.')

        self.comment = self.getComment(data)
        prefix = data['level1'].file.filename.split('/')[-1].split('.hd5')[0]
        self.outfilename = '{}/{}_Level2RRL.hd5'.format(self.output_dir,prefix)

        # Skip files if we aren't overwriting
        if not self.overwrite: 
            return data
        data = self.setReadWrite(data)
        self.logger(f'{fname}:{self.name}: Applying vane calibration.')
        self.run(data['level1'].file)
        self.logger(f'{fname}:{self.name}: Writing level 2 file: {self.outfilename}')
        # Want to ensure the data file is read/write
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
            cname = '{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,fname)
            gain = np.ones((2, nHorns, nSBs, nChans))
            tsys = np.ones((2, nHorns, nSBs, nChans))

        # Future: Astro Calibration
        # if self.cal_mode.upper() == 'ASTRO':
        # gain = self.getcalibration_astro(data)
        self.output = np.zeros((nHorns,len(self.rrl_frequencies),11,nSamples))
        self.spectra  = np.zeros((len(self.feeds),len(self.rrl_qnumbers),21,nSamples))*np.nan
        self.velocity = np.zeros((len(self.feeds),len(self.rrl_qnumbers),21))*np.nan
        self.frequency = np.zeros((len(self.feeds),len(self.rrl_qnumbers),21))*np.nan
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
                    
                        self.spectra[ifeed,iqno,:(hi-lo),:] = s1[lo:hi]
                        self.velocity[ifeed,iqno,:(hi-lo)]  = velocity[lo:hi]
                        self.frequency[ifeed,iqno,:(hi-lo)]  = frequencies[sb,lo:hi]

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 RRL file with an external link to the original level 1 data
        """
        if os.path.exists(self.outfilename):
            self.outfile = h5py.File(self.outfilename,'a')
        else:
            self.outfile = h5py.File(self.outfilename,'w')

        # Set permissions and group
        if self.set_permissions:
            try:
                os.chmod(self.outfilename,0o664)
                shutil.chown(self.outfilename, group=self.permissions_group)
            except PermissionError:
                self.logger(f'{self.name}: Warning, couldnt set the file permissions.')

        if 'tod' in self.outfile:
            del self.outfile['tod']
        tod_dset = self.outfile.create_dataset('tod',data=self.spectra, dtype=self.spectra.dtype)
            
        tod_dset.attrs['Unit'] = 'K'
        tod_dset.attrs['Calibration'] = '{self.cal_mode}:{self.cal_prefix}'

        if 'velocity' in self.outfile:
            del self.outfile['velocity']
        tod_dset = self.outfile.create_dataset('velocity',data=self.velocity, dtype=self.velocity.dtype)
        tod_dset.attrs['Unit'] = 'km/s'

        if 'frequency' in self.outfile:
            del self.outfile['frequency']
        freq_dset = self.outfile.create_dataset('frequency',data=self.rrl_frequencies, dtype=self.rrl_frequencies.dtype)

        if 'feeds' in self.outfile:
            del self.outfile['feeds']
        freq_dset = self.outfile.create_dataset('feeds',data=np.array(self.feeds), dtype=np.array(self.feeds).dtype)

        self.outfile.attrs['version'] = __level2_version__

        # Add version info
        self.outfile.attrs['pipeline-version'] = comancpipeline.__version__

        # Link the rrl data to level2
        data_filename = data.filename
        fname = data.filename.split('/')[-1]
        
        # Link to the level2 continuum file
        if self.level2 in data.keys():
            del data[self.level2]
        data[self.level2] = h5py.ExternalLink(self.outfilename,'/')
        
        




class CompareTsys(BaseClasses.DataStructure):
    """
    Compare the Tsys from the Vane with the RMS of auto subtraction
    """

    def __init__(self, feeds='all', 
                 output_obsid_starts = [0],
                 output_obsid_ends   = [None],
                 output_dirs = ['.'],
                 nworkers= 1,
                 database=None,
                 average_width=512,
                 calvanedir='CalVanes',
                 cal_mode = 'Vane', 
                 vaneprefix='VaneCal',
                 level2='level2',
                 data_dirs=None,
                 set_permissions=True,
                 mask_width=20,
                 permissions_group='comap',
                 calvane_prefix='CalVane',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)

        self.name = 'CompareTsys'
        self.feeds_select = feeds

        # Channels to mask either side of tsys spike 
        self.mask_width = mask_width

        # We will be writing level 2 data to multiple drives,
        #  drive to write to will be set by the obsid of the data
        self.output_dirs = output_dirs
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends

        if isinstance(data_dirs,list):
            self.data_dirs = data_dirs
        else:
            self.data_dirs = [data_dirs]

        self.nworkers = int(nworkers)
        self.average_width = int(average_width)

        self.calvanedir = calvanedir
        self.calvane_prefix = calvane_prefix

        self.cal_mode = cal_mode
        self.vaneprefix=vaneprefix

        self.level2=level2
        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

        self.database   = database + '_{}'.format(os.get_pid())

    def __str__(self):
        return "Running {}".format(self.name)

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
        self.nFeeds, self.nBands, self.nChannels,self.nSamples = data['spectrometer/tod'].shape

        frequency = data['spectrometer/frequency'][...]
        features = np.log(self.getFeatures(data))/np.log(2)
        gd = (features != 13) & np.isfinite(features)
        cname, gain, tsys,spikes = self.getcalibration_obs(data)

        self.mask = np.zeros((self.nFeeds,self.nBands, self.nChannels),dtype=bool)
        self.peak_freqs= np.empty((self.nFeeds, self.nBands),dtype=object)
        self.peak_amps = np.empty((self.nFeeds, self.nBands),dtype=object)
        for ifeed,feed in enumerate(self.feeds):
            for iband in range(self.nBands):
                self.mask[ifeed,iband], self.peak_amps[ifeed,iband], self.peak_freqs[ifeed,iband] = self.find_tsys_spikes(tsys[0,ifeed,iband,:],frequency[iband])
                
        tsamp = float(data['comap'].attrs['tsamp'])
        bw = 2e9/1024

        self.tsys_rms = np.zeros((self.nFeeds, self.nBands, self.nChannels))
        self.auto_rms = np.zeros((self.nFeeds, self.nBands, self.nChannels))
        for ifeed, feed in enumerate(tqdm(self.feeds,desc=self.name)):
            d = data['spectrometer/tod'][ifeed,...] 
            d = d[...,gd]
            N = int(d.shape[-1]//2*2)
        
            d = d/gain[0,ifeed,...,None]
            self.tsys_rms[ifeed] = tsys[0,ifeed]/np.sqrt(bw*tsamp)
            self.auto_rms[ifeed] = np.nanstd(d[...,1:N:2]-d[...,0:N:2],axis=-1)/np.sqrt(2)

    def find_tsys_spikes(self,tsys,frequency):
        chans = np.arange(tsys.size)
        peaks,properties = find_peaks(tsys,prominence=3,width=[2,60])
        gd = ((chans > 4) & (chans < tsys.size-4))
        gd [tsys.size//2-2:tsys.size//2+2] = False

        mask = np.ones(tsys.size,dtype=bool)
        mask[~gd] = False
        if len(peaks) == 0:
            return mask, np.empty(0), np.empty(0)
        widths = (properties['right_ips']-properties['left_ips'])*2
        bad = np.zeros(tsys.size,dtype=bool)   

        for ipeak,c in enumerate(peaks):
            mask_width = np.max([widths[ipeak],self.mask_width])
            sel = (np.abs(chans-c) < self.mask_width)
            mask[sel] = False
        return mask, tsys[peaks], frequency[peaks]

    def okay_level2_version(self,h):
        """
        Check level 2 file is up to date
        """
        if not self.level2 in h:
            return False

        if not 'version' in h[self.level2].attrs:
            return False
        print( h[self.level2].attrs['version'], __level2_version__)
        if not h[self.level2].attrs['version'] == __level2_version__:
            return False
        if not 'vane-version' in h[self.level2].attrs:
            return False
        if not h[self.level2].attrs['vane-version'] == h[self.level2]['Vane'].attrs['version']:
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

        self.obsid = self.getObsID(data)
        self.comment = self.getComment(data)
        prefix = data.filename.split('/')[-1]

        self.output_dir = self.getOutputDir(self.obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)
        self.calvanepath = '{}/{}'.format(self.output_dir, self.calvanedir)


        self.outfilename = '{}/{}_{}'.format(self.calvanepath,self.vaneprefix,fname)
        

        # Skip files that are already calibrated:        
        if os.path.exists(self.outfilename) & (not self.overwrite): 
            self.logger(f'{fname}:{self.name}: Level 2 file exists...checking vane version...')
            self.outfile = h5py.File(self.outfilename,'r')
            if self.level2 in self.outfile:
                if self.okay_level2_version(self.outfile):
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
        if not isinstance(self.database,type(None)):
            self.write_database(data)

        return data

    def getcalibration_obs(self,data):
        """
        Open vane calibration file.
        """
        fname = data.filename.split('/')[-1]
        cname = '{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,fname)
        gain_file = h5py.File(cname,'r')
        gain = gain_file['Gain'][...] # (event, horn, sideband, frequency)
        tsys = gain_file['Tsys'][...] # (event, horn, sideband, frequency) - use for weights
        spikes = gain_file['Spikes'][...]
        print(gain.shape,tsys.shape,spikes.shape)
        gain_file.close()
        return cname, gain, tsys, spikes

    def write_database(self,data):
        """
        Write the fits to the general database
        """

        if not os.path.exists(self.database):
            output = FileTools.safe_hdf5_open(self.database,'w')
        else:
            output = FileTools.safe_hdf5_open(self.database,'a')

        obsid = self.getObsID(data)
        if obsid in output:
            ogrp = output[obsid]
        else:
            ogrp = output.create_group(obsid)

        if self.name in ogrp:
            lvl2 = ogrp[self.name]
        else:
            lvl2 = ogrp.create_group(self.name)


        # Store datasets in root
        dnames = ['tsys_rms','auto_rms','Spikes']
        dsets  = [self.tsys_rms, 
                  self.auto_rms, 
                  self.mask]
        for (dname, dset) in zip(dnames, dsets):
            if dname in lvl2:
                del lvl2[dname]
            lvl2.create_dataset(dname,  data=dset)



        # Store the peak frequencies and amplitudes
        grpname = 'SpikeInfo'
        if grpname in lvl2:
            grp = lvl2[grpname]
        else:
            grp = lvl2.create_group(grpname)
        dnames = ['Amplitudes','Frequencies']
        dsets  = [self.peak_amps, self.peak_freqs]
        self.feeds,_,_ = self.getFeeds(data,'all')
        for (_dname, _dset) in zip(dnames, dsets):
            for ifeed, feed in enumerate(self.feeds):
                dname = '{}_Feed{:02d}'.format(_dname,feed)  
                dset = np.concatenate(_dset[ifeed])
                if dname in lvl2:
                    del lvl2[dname]
                lvl2.create_dataset(dname,  data=dset)

        
        output.close()

    def write(self,data):
        """
        Add new spikes + rms into VaneCal file
        """
        if os.path.exists(self.outfilename):
            output = h5py.File(self.outfilename,'a')
        else:
            output = h5py.File(self.outfilename,'w')

        # Set permissions and group
        if self.set_permissions:
            try:
                os.chmod(self.outfilename,0o664)
                shutil.chown(self.outfilename, group=self.permissions_group)
            except PermissionError:
                self.logger(f'{self.name}: Warning, couldnt set the file permissions.')



        # Store datasets in root
        dnames = ['tsys_rms','auto_rms','Spikes']
        dsets  = [self.tsys_rms, 
                  self.auto_rms, 
                  self.mask]
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)



        # Store the peak frequencies and amplitudes
        grpname = 'SpikeInfo'
        if grpname in output:
            grp = output[grpname]
        else:
            grp = output.create_group(grpname)
        dnames = ['Amplitudes','Frequencies']
        dsets  = [self.peak_amps, self.peak_freqs]
        self.feeds,_,_ = self.getFeeds(data,'all')
        for (_dname, _dset) in zip(dnames, dsets):
            for ifeed, feed in enumerate(self.feeds):
                dname = '{}_Feed{:02d}'.format(_dname,feed)  
                dset = np.concatenate(_dset[ifeed])
                if dname in output:
                    del output[dname]
                output.create_dataset(dname,  data=dset)
