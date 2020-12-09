import numpy as np
from matplotlib import pyplot
import h5py
import comancpipeline 
from comancpipeline.Analysis.BaseClasses import DataStructure
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
import pandas as pd
import os
from astropy.time import Time
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

class NoColdError(Exception):
    pass
class NoHotError(Exception):
    pass
class NoDiodeError(Exception):
    pass


class CreateLevel2Cont(DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all', output_dir='', nworkers= 1, 
                 average_width=512,calvanedir='AncillaryData/CalVanes',
                 calvane_prefix='CalVane'):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """

        self.feeds_select = feeds

        self.output_dir = output_dir

        self.nworkers = int(nworkers)
        self.average_width = int(average_width)
        
        self.calvanedir = calvanedir
        self.calvane_prefix = calvane_prefix

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
        if self.feeds_select == 'all':
            feeds = data['spectrometer/feeds'][:]
        else:
            if (not isinstance(self.feeds_select,list)) & (not isinstance(self.feeds_select,np.ndarray)) :
                self.feeds = [int(self.feeds_select)]
                feeds = self.feeds
            else:
                feeds = [int(f) for f in self.feeds_select]

        self.feeds = feeds
        self.feedlist = data['spectrometer/feeds'][:]
        self.feeddict = {feedid:feedindex for feedindex, feedid in enumerate(self.feedlist)}


        # Setup output file here, we will have to write in sections.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if os.path.isfile(self.outfilename):
            os.remove(self.outfilename)


        # Opening file here to write out data bit by bit
        avg_tod_shape = (alltod.shape[0], alltod.shape[1], alltod.shape[2]//self.average_width, alltod.shape[3])
        self.avg_tod = np.zeros(avg_tod_shape,dtype=alltod.dtype)

        # Average the data and apply the gains
        self.average(data.filename,data,alltod, self.avg_tod)


    def __call__(self,data):
        """
        Modify baseclass __call__ to change file from the level1 file to the level2 file.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        if 'comment' in data['comap'].attrs:
            comment = data['comap'].attrs['comment']
            if not isinstance(comment,str):
                comment = comment.decode('utf-8')
        else:
            comment = 'No Comment'

        prefix = data.filename.split('/')[-1].split('.hd5')[0]
        self.outfilename = '{}/{}_Level2Cont.hd5'.format(self.output_dir,prefix)

        if os.path.exists(self.outfilename): # skip if we have already calibrated
            self.outfile = h5py.File(self.outfilename,'a')
            lvl2 = self.outfile['level2']
            if 'pipeline-version' in lvl2.attrs:
                if lvl2.attrs['pipeline-version'] == comancpipeline.__version__:
                    data.close()
                    data = self.outfile
                    return data
                else:
                    self.outfile.close()
            else:
                self.outfile.close()


        # Want to check the versions
        if os.path.isfile(self.outfilename):
            h = h5py.File(self.outfilename,'r')
            if 'pipeline-version' in h['level2'].attrs.keys(): # Want to skip files that are already processed
               if  h['level2'].attrs['pipeline-version'] == comancpipeline.__version__:
                   h.close()
                   return data
            h.close()

        # Ignore Sky dips
        if 'Sky nod' in comment:
            return data

        self.run(data)
        self.write(data)

        # Now change to the level2 file for all future analyses.
        data.close()
        data = self.outfile

        return self.outfile

    def average(self,filename,data, alltod, tod):
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
            fname = data.filename.split('/')[-1]
            cname = '{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname)
            gain_file = h5py.File(cname,'r')
            gain = gain_file['Gain'] # (event, horn, sideband, frequency) 
            tsys = gain_file['Tsys'] # (event, horn, sideband, frequency) - use for weights
        except ValueError:
            cname = '{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname)
            gain = np.ones((2, nHorns, nSBs, nChans*width))
            tsys = np.ones((2, nHorns, nSBs, nChans*width))

        for ifeed, feed in enumerate(self.feeds):
            feed_array_index = self.feeddict[feed]
            for sb in tqdm(range(nSBs)):
                # Weights/gains already snipped to just the feeds we want
                w, g = 1./tsys[0,ifeed, sb, :]**2, gain[0,ifeed, sb, :]

                gvals = np.zeros(nChans)
                for chan in range(nChans):
                    try:
                        bot = np.nansum(w[chan*width:(chan+1)*width])
                    except:
                        continue

                    d = alltod[feed_array_index,sb,chan*width:(chan+1)*width,:]
                    caltod = d/g[chan*width:(chan+1)*width,np.newaxis]

                    if width > 1:
                        self.avg_tod[ifeed,sb,chan,:] = np.sum(caltod*w[chan*width:(chan+1)*width,np.newaxis],axis=0)/bot
                    else:
                        self.avg_tod[ifeed,sb,chan,:] = caltod


                    self.avg_frequency[sb,chan] = np.mean(frequency[sb,chan*width:(chan+1)*width])
        gain_file.close()

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


        if 'level2' in self.outfile:
            del self.outfile['level2']
        lvl2 = self.outfile.create_group('level2')

        tod_dset = lvl2.create_dataset('averaged_tod',data=self.avg_tod, dtype=self.avg_tod.dtype)
        tod_dset.attrs['Unit'] = 'K'
        tod_dset.attrs['Calibration'] = 'Vane'

        freq_dset = lvl2.create_dataset('frequency',data=self.avg_frequency, dtype=self.avg_frequency.dtype)

        # Link the Level1 data
        self.outfile['level1'] = h5py.ExternalLink(data.filename,'/')

        # Add version info
        lvl2.attrs['pipeline-version'] = comancpipeline.__version__

        # Link the Level1 data
        fname = data.filename.split('/')[-1]
        lvl2['Vane'] = h5py.ExternalLink('{}/{}_{}'.format(self.calvanedir,self.calvane_prefix,fname),'/')

class CalculateVaneMeasurement(DataStructure):
    """
    Calculates the Tsys and Gain factors from a COMAP vane in/out measurement.
    """
    def __init__(self, output_dir='AmbientLoads/', 
                 overwrite=True, elLim=5, feeds = 'all',
                 minSamples=200, tCold=2.74, tHotOffset=273.15,prefix='VaneCal'):
        super().__init__()
        self.feeds_select = feeds 

        self.output_dir = output_dir
        self.overwrite= overwrite
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
        
        if 'comment' in data['comap'].attrs:
            comment = data['comap'].attrs['comment']
            if not isinstance(comment,str):
                comment = comment.decode('utf-8')
        else:
            comment = 'No Comment'

        # Ignore Sky dips
        if 'Sky nod' in comment:
            return data


        self.run(data)
        self.write(data)

        return data

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
        return idHot, idCold

    def FindVane(self,data):
        """
        Find the calibration vane start and end positions.
        Raise NoDiodeError if no vane found
        """
        mjd  = data['spectrometer/MJD']

        # Need to count how many calvane events occur, look for features 2**13
        if mjd[0] > Time(datetime(2019,6,14),format='datetime').mjd: # The cal vane feature bit can be trusted after 14 June 2019
            features = np.floor(np.log(data['spectrometer/features'][:])/np.log(2)).astype(int)
            justVanes = np.where((features == 13))[0]
        else: # Must use cal vane angle to calculate the diode positions

            if mjd[0] < Time(datetime(2019,3,1),format='datetime').mjd: # Early observations before antenna0 
                hkMJD = data['hk/vane/MJD'][:]
                angles = np.interp(mjd,hkMJD, data['hk/vane/angle'][:])
            else:
                hkMJD = data['hk/antenna0/vane/utc'][:]
                angles = np.interp(mjd,hkMJD, data['hk/antenna0/vane/angle'][:])

            justVanes = np.where((angles < 21000))[0]

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

        return vanePositions

    def run(self,data):
        """
        Calculate the cal vane calibration factors for observations after
        June 14 2019 (MJD: 58635) after which corrections to the calvane features
        were made.

        """

        fname = data.filename.split('/')[-1]
        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)
        if os.path.exists(outfile): # If the vane has already be calculated, skip
            self.nodata=True
            return data

        # Read in data that is required:
        freq = data['spectrometer/frequency'][...]
        mjd  = data['spectrometer/MJD'][:] 
        el  = data['pointing/elActual'][:]        

        if self.feeds_select == 'all':
            feeds = data['spectrometer/feeds'][:]
        else:
            if (not isinstance(self.feeds_select,list)) & (not isinstance(self.feeds_select,np.ndarray)) :
                self.feeds = [int(self.feeds_select)]
                feeds = self.feeds
            else:
                feeds = [int(f) for f in self.feeds_select]
        self.feeds = feeds

        self.mjd = np.mean(mjd)
        self.elevation = np.nanmedian(el)

        # Keep TOD as a file object to avoid reading it all in
        tod  = data['spectrometer/tod'] 
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

        vanePositions = self.FindVane(data)
        nVanes = len(vanePositions)

        self.vane_samples = np.zeros((nVanes,2))
        # Create output containers
        self.Tsys = np.zeros((nVanes, nHorns, nSBs, nChan))
        self.Gain = np.zeros((nVanes, nHorns, nSBs, nChan))
        self.RMS  = np.zeros((nVanes, nHorns, nSBs, nChan))


        # Now loop over each event:
        for horn, feedid in enumerate(tqdm(self.feeds)):
            for vane_event, (start, end) in enumerate(vanePositions):
                # Get the mean time of the event
                if horn == 0:
                    self.vane_samples[vane_event,:] = int(start), int(end)

                tod_slice = tod[horn,:,:,start:end]

                btod_slice = btod[horn,:,start:end]

                try:
                    idHot, idCold = self.findHotCold(btod_slice[0,:])
                except (NoHotError,NoColdError) as e:
                    print(e)
                    break


                time= np.arange(tod_slice.shape[-1])
                for sb in range(nSBs):
                    vHot = np.nanmedian(tod_slice[sb,:,idHot],axis=0)
                    vCold= np.nanmedian(tod_slice[sb,:,idCold],axis=0)
                    Y = vHot/vCold
                    self.Tsys[vane_event,horn,sb,:] = ((tHot - self.tCold)/(Y - 1.) ) - self.tCold
                    self.Gain[vane_event,horn,sb,:] = ((vHot - vCold)/(tHot - self.tCold))
                    tod_slice[sb,:,:] /= self.Gain[vane_event,horn,sb,:,np.newaxis]


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
        dnames = ['Tsys','Gain','VaneEdges']
        dsets  = [self.Tsys, self.Gain, self.vane_samples]
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)

        output['Tsys'].attrs['Unit'] = 'K'
        output['Gain'].attrs['Unit'] = 'V/K'
        output['VaneEdges'].attrs['Unit'] = 'Index'
        output.close()
