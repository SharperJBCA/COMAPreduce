# Routines for downsampling the level 2 data to wide band chunks for continuum data analysis
# Data will be cleaned such that it is ready to enter the destriper:
# 1) From Statistics : Remove atmosphere and baselines 
# 2) From astrocalibration : Apply Jupiter calibration to each feed

import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
from comancpipeline.Tools import  binFuncs, stats, FileTools

from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting
from comancpipeline.Analysis import Statistics
from scipy import signal

import time
import os

import shutil

from tqdm import tqdm

__level3_version__='v062022'

def subtract_filters(tod,az,el,filter_tod, filter_coefficients, atmos, atmos_coefficient):
    """
    Return the TOD with median filter and atmosphere model subtracted
    """
    tod_out = tod - filter_tod*filter_coefficients -\
              Statistics.AtmosGroundModel(atmos,az,el)*atmos_coefficient
    tod_out -= np.nanmedian(tod_out)
    return tod_out 

class CreateLevel3(BaseClasses.DataStructure):
    def __init__(self,
                 level2='level2',
                 level3='level3',
                 database=None, 
                 output_obsid_starts = [0],
                 output_obsid_ends   = [None],
                 output_dirs = ['.'],
                 cal_source='taua',
                 set_permissions=True,
                 permissions_group='comap',
                 median_filter=True,
                 atmosphere=True,
                 astro_cal=True, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.name = 'CreateLevel3'

        self.output_dirs = output_dirs
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends

        self.database = database

        self.level2=level2
        self.level3=level3
        self.cal_source=cal_source

        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

        self.astro_cal=astro_cal
        self.median_filter = median_filter
        self.atmosphere = atmosphere
    def __str__(self):
        return "Creating Level 3"

    def __call__(self,data):
        """
        """

        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        data_dir = data.filename.split(fname)[0]

        # obtain obsid
        obsid = int(data.filename.split('/')[-1].split('-')[1])
        # determine output directory
        self.output_dir = self.getOutputDir(obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)

        self.outfile = '{}/{}_{}'.format(self.output_dir,self.level3,fname)

        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        comment = self.getComment(data)

        if 'Sky nod' in comment:
            return data

        if not self.level2 in data.keys():
            self.logger(f'{fname}:{self.name}:Error: No {self.level2} data found?')
            return data

        if not 'Statistics' in data[self.level2].keys():
            self.logger(f'{fname}:{self.name}:Error: No {self.level2}/Statistics found?')
            return data

        if not 'scan_edges' in data[f'{self.level2}/Statistics'].keys():
            self.logger(f'{fname}:{self.name}:Error: No {self.level2}/Statistics/scan_edges found?')
            return data

        

        if os.path.exists(self.outfile) & (not self.overwrite):
            self.logger(f'{fname}:{self.name}: {self.level3}_{fname} exists, ignoring (overwrite = {self.overwrite})')
            return data
        

        self.logger(f'{fname}:{self.name}: Creating {self.level3} data.')
        self.run(data)

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Writing to {self.outfile}')
        self.write(data)
        self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def create_simulation(self,data):
        """
        """

        self.all_tod = data['level2/averaged_tod'][...]
        self.all_tod = np.reshape(self.all_tod, (self.all_tod.shape[0],
                                                 self.all_tod.shape[1]*self.all_tod.shape[2],
                                                 self.all_tod.shape[3]))
        N = int(self.all_tod.shape[-1]//2*2)
        rms = np.nanstd(self.all_tod[:,:,:N:2] - self.all_tod[:,:,1:N:2],axis=-1)/np.sqrt(2)
        rms = rms[...,None]*np.ones(self.all_tod.shape[-1])[None,None,:]
        self.all_weights = 1./rms**2
        self.cal_factors = self.all_tod*0.+1
        self.all_frequency = data['level2/frequency'][...].flatten()


    def get_cal_gains(self,this_obsid):
        """
        Get all of the gain factors associated with this calibration source
        """

        db = FileTools.safe_hdf5_open(self.database,'r')
        obsids = []
        for obsid, grp in db.items():
            if not 'Flagged' in grp.attrs:
                continue
            if grp.attrs['Flagged']:
                continue
            if not 'TauA' in grp['level2'].attrs['source']:
                continue

            obsids += [int(obsid)]

        obsids = np.array(obsids)
        idx = np.argmin(np.abs(obsids-this_obsid))
        self.nearest_calibrator = str(obsids[idx])
        gains = {'Gains':db[self.nearest_calibrator]['FitSource/Gains'][...],
                 'Feeds':db[self.nearest_calibrator]['FitSource/feeds'][...]}

        db.close()

        return gains

    def get_channel_mask(self,this_obsid,feed):
        """
        Get the updated channel mask for this obsid
        """

        db = FileTools.safe_hdf5_open(self.database,'r')
        channel_mask = db[str(this_obsid)]['Vane/Level2Mask'][feed-1,...]
        db.close()
        return channel_mask

    def calibrate_tod(self,data, tod, weights, ifeed, feed):
        """
        Calibrates data using a given calibration source
        """
        this_obsid = int(self.getObsID(data))
        # Get Gain Calibration Factors
        gains = self.get_cal_gains(this_obsid)

        nbands, nchan, ntod = tod.shape
        if self.cal_source != 'none':
            idx = np.where((gains['Feeds'] == feed))[0]
            if not len(idx) == 0:
                cal_factors = gains['Gains'][idx,...]
            else:
                cal_factors = np.ones((1,tod.shape[0],tod.shape[1]))

        tod = tod/cal_factors[0,...,None]
        weights = weights*cal_factors[0,...,None]**2
        bad = ~np.isfinite(tod) | ~np.isfinite(weights)
        tod[bad] = 0
        weights[bad] = 0


        channel_mask = self.get_channel_mask(this_obsid,feed)
        tod[~channel_mask] = 0
        weights[~channel_mask] = 0

        return tod, weights, cal_factors[0]

    def clean_tod(self,d,ifeed,feed):
        """
        Subtracts the median filter and atmosphere from each channel per scan
        """
        scan_edges = d[f'{self.level2}/Statistics/scan_edges'][...]
        nscans = scan_edges.shape[0]

        feed_tod = d[f'{self.level2}/averaged_tod'][ifeed,:,:,:]
        weights  = np.zeros(feed_tod.shape)
        mask     = np.zeros(feed_tod.shape[-1],dtype=bool)
        az = d['level1/spectrometer/pixel_pointing/pixel_az'][ifeed,:]
        el = d['level1/spectrometer/pixel_pointing/pixel_el'][ifeed,:]

        # Statistics for this feed                        
        medfilt_coefficient = d[f'{self.level2}/Statistics/filter_coefficients'][ifeed,...]
        atmos = d[f'{self.level2}/Statistics/atmos'][ifeed,...]
        atmos_coefficient = d[f'{self.level2}/Statistics/atmos_coefficients'][ifeed,...]
        wnoise_auto = d[f'{self.level2}/Statistics/wnoise_auto'][ifeed,...]
        fnoise_fits = d[f'{self.level2}/Statistics/fnoise_fits'][ifeed,...]

        # then the data for each scan
        last = 0
        scan_samples = []
        for iscan,(start,end) in enumerate(tqdm(scan_edges)):
            scan_samples = np.arange(start,end,dtype=int)
            median_filter = d[f'{self.level2}/Statistics/FilterTod_Scan{iscan:02d}'][ifeed,...]
            N = int((end-start))
            end = start+N
            tod = feed_tod[...,start:end]
            mask[start:end] = True
            # Subtract atmospheric fluctuations per channel
            for iband in range(4):
                for ichannel in range(64):
                    #if self.channelmask[ifeed,iband,ichannel] == False:
                    amdl = Statistics.AtmosGroundModel(atmos[iband,iscan],az[start:end],el[start:end]) *\
                           atmos_coefficient[iband,ichannel,iscan,0]
                    if self.median_filter:
                        tod[iband,ichannel,:] -= median_filter[iband,:N] * medfilt_coefficient[iband,ichannel,iscan,0]
                    if self.atmosphere:
                        tod[iband,ichannel,:] -= amdl
                    tod[iband,ichannel,:] -= np.nanmedian(tod[iband,ichannel,:])


            wnoise = wnoise_auto[:,:,iscan,0]
            weights[...,start:end] = 1./wnoise[...,None]**2
            bad = np.isnan(weights) | np.isinf(weights) | ~np.isfinite(feed_tod)
            feed_tod[bad] = 0
            weights[bad] = 0

        return feed_tod, weights, mask

    def average_tod(self,d,feed_tod,feed_weights,mask):
        """
        Average together to TOD 
        """        

        frequency = d['level1/spectrometer/frequency'][...]
        # This becomes (nBands, 64)        
        self.frequency = np.mean(np.reshape(frequency,(frequency.shape[0],frequency.shape[1]//16,16)) ,axis=-1)
        all_tod = np.zeros((8, feed_tod.shape[-1]))
        all_weights=np.zeros((8, feed_tod.shape[-1]))
        all_frequency=np.zeros((8))
        for ichan, (flow,fhigh) in enumerate(zip(np.arange(8)+26,np.arange(8)+27)):
            sel = ((self.frequency >= flow) & (self.frequency < fhigh))
            top = np.sum(feed_tod[sel,:]*feed_weights[sel,:],axis=0)
            bot = np.sum(feed_weights[sel,:],axis=0)
            all_tod[ichan,:] = top/bot
            all_weights[ichan,:] = bot
            all_frequency[ichan] = (fhigh+flow)/2.
            
        diff = all_tod[:,mask]
        N = int(diff.shape[1]//2*2)
        diff = (diff[:,:N:2]-diff[:,1:N:2])
        auto = stats.MAD(diff.T)

        amean_rms = np.sqrt(1./np.nanmedian(all_weights[:,mask],axis=1))

        # Add the weighted average uncertainties to the auto-rms uncertainties
        all_weights = 1./(1./all_weights + auto[:,None]**2)

        return all_tod, all_weights, auto, all_frequency
        
    def run(self, d):
        """
        Expects a level2 file structure to be passed.
        """

        feeds,feedidx,_ = self.getFeeds(d,'all')

        tod_shape = d[f'{self.level2}/averaged_tod'].shape
        
        scanedges = d[f'{self.level2}/Statistics/scan_edges'][...]
        nfeeds = 20
        nchannels = 8
        
        self.all_tod       = np.zeros((20, nchannels, tod_shape[-1])) 
        self.all_weights   = np.zeros((20, nchannels, tod_shape[-1])) 
        self.all_frequency = np.zeros((nchannels)) 
        self.all_auto = np.zeros((20,nchannels)) 
        self.all_mask = np.zeros((20,tod_shape[-1]))
        self.all_cal_factors = np.zeros((20,4,64))
        # Read in data from each feed
        for ifeed,feed in enumerate(tqdm(feeds,desc='Looping over feeds')):
            if feeds[ifeed] == 20:
                continue
            feed_tod,feed_weights,mask = self.clean_tod(d,ifeed,feed)

            if self.astro_cal:
                feed_tod,feed_weights,cal_factors  = self.calibrate_tod(d,feed_tod,feed_weights,ifeed,feed)
            else:
                cal_factors = 1

            self.all_tod[feed-1],self.all_weights[feed-1], self.all_auto[feed-1], self.all_frequency = self.average_tod(d,feed_tod,feed_weights,mask) 
            self.all_mask[feed-1] = mask
            self.all_cal_factors[feed-1] = cal_factors
        
    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # We will store these in a separate file and link them to the level2s
        fname = data.filename.split('/')[-1]
        
        if os.path.exists(self.outfile):
            output = h5py.File(self.outfile,'a')
        else:
            output = h5py.File(self.outfile,'w')

        # Set permissions and group
        if self.set_permissions:
            try:
                os.chmod(self.outfile,0o664)
                shutil.chown(self.outfile, group=self.permissions_group)
            except PermissionError:
                self.logger(f'{fname}:{self.name}: Warning, couldnt set the file permissions.')

        # Store datasets in root
        data_out = {'tod':self.all_tod,
                    'weights':self.all_weights,
                    'mask':self.all_mask,
                    'cal_factors':self.all_cal_factors,
                    'frequency':self.all_frequency,
                    'auto_rms':self.all_auto}

        for dname, dset in data_out.items():
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)

        output.attrs['version'] = __level3_version__
        output['cal_factors'].attrs['source'] = self.cal_source
        output['cal_factors'].attrs['calibrator_obsid'] = self.nearest_calibrator

        output.close()
        
        if self.level3 in data.keys():
            del data[self.level3]
        data[self.level3] = h5py.ExternalLink(self.outfile,'/')

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

        grp.attrs['level3_filename'] = self.outfile

        if self.name in grp:
            del grp[self.name]
        lvl3 = grp.create_group(self.name)

        lvl3.attrs['version'] = __level3_version__
        lvl3.attrs['calibrator_obsid'] = self.nearest_calibrator
        lvl3.attrs['calibrator_source'] = self.cal_source
        output.close()
