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
from comancpipeline.Tools import  binFuncs, stats

from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting
from comancpipeline.Analysis import Statistics
from scipy import signal

import time
import os

import shutil

from tqdm import tqdm
from comancpipeline.data import Data

__level3_version__='v2'

class CreateLevel3(BaseClasses.DataStructure):
    def __init__(self,level2='level2',level3='level3',output_dir = None,cal_source='taua',
                 set_permissions=True,
                 permissions_group='comap',
                 channel_mask=None, gain_mask=None, calibration_factors=None, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.name = 'CreateLevel3'
        # READ ANY ANCILLARY DATA: MASKS/CALIBRATION FACTORS
        if not isinstance(channel_mask,type(None)):
            self.channelmask = np.load(channel_mask,allow_pickle=True).astype(bool)
        else:
            self.channelmask = None

        if not isinstance(gain_mask,type(None)):
            self.gainmask = np.load(gain_mask,allow_pickle=True).astype(bool)
        else:
            self.gainmask = None

        if not isinstance(calibration_factors, type(None)):
            self.calfactors =  np.load(calibration_factors)
        else:
            self.calfactors = None

        self.output_dir = output_dir

        self.level2=level2
        self.level3=level3
        self.cal_source=cal_source

        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

    def __str__(self):
        return "Creating Level 3"

    def __call__(self,data):
        """
        """

        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        data_dir = data.filename.split(fname)[0]
        if isinstance(self.output_dir,type(None)):
            self.output_dir = f'{data_dir}/{self.level3}'
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
        self.calibrate_data(data)

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Writing to {self.outfile}')
        self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data

    def calibrate_data(self,data):
        """
        Calibrates data using a given calibration source
        """
        feeds, feed_indices, feed_dict = self.getFeeds(data,'all')
        this_obsid = int(self.getObsID(data))
        # Get Gain Calibration Factors

        nfeeds, nchan, ntod = self.all_tod.shape
        self.cal_factors = np.zeros((nfeeds,nchan))
        for ifeed,feed_num in enumerate(feeds):
            obsids = Data.feed_gains[self.cal_source.lower()]['obsids']*1
            gains  = Data.feed_gains[self.cal_source.lower()]['gains'][:,feed_num-1,:,:]

            # now find the nearest non-nan obsid to calibrate off
            obs_idx = np.argmin((obsids - this_obsid)**2)
            self.cal_factors[ifeed,...] = gains[obs_idx].flatten()
        self.all_tod = self.all_tod/self.cal_factors[:,:,None]
        self.all_weights = self.all_weights*self.cal_factors[:,:,None]**2



    def run(self, d):
        """
        Expects a level2 file structure to be passed.
        """
        tod_shape = d[f'{self.level2}/averaged_tod'].shape

        scan_edges = d[f'{self.level2}/Statistics/scan_edges'][...]
        nscans = scan_edges.shape[0]
        nchannels = 8
        self.all_tod       = np.zeros((tod_shape[0], nchannels, tod_shape[-1])) 
        self.all_weights   = np.zeros((tod_shape[0], nchannels, tod_shape[-1])) 
        self.all_frequency = np.zeros((nchannels)) 
        frequency = d['level1/spectrometer/frequency'][...]
        self.frequency = np.mean(np.reshape(frequency,(frequency.shape[0],frequency.shape[1]//16,16)) ,axis=-1).flatten()
        feeds = d['level1/spectrometer/feeds'][...]
        feedids = np.concatenate([np.arange(8).astype(int)+i*8 for i in range(20) if i+1 in feeds])
        xfeed,yfeed = np.meshgrid(feedids,feedids)
        self.correlation_matrix = np.zeros((nscans,20*8,20*8))
        # Read in data from each feed

        for index, ifeed in enumerate(range(tod_shape[0])):
            if feeds[ifeed] == 20:
                continue
            todin = d[f'{self.level2}/averaged_tod'][ifeed,:,:,:]
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
            for iscan,(start,end) in enumerate(scan_edges):
                scan_samples = np.arange(start,end,dtype=int)
                median_filter = d[f'{self.level2}/Statistics/FilterTod_Scan{iscan:02d}'][ifeed,...]
                N = int((end-start))
                end = start+N
                tod = todin[...,start:end]

                # Subtract atmospheric fluctuations per channel
                for iband in range(4):
                    for ichannel in range(64):
                        #if self.channelmask[ifeed,iband,ichannel] == False:
                        amdl = Statistics.AtmosGroundModel(atmos[iband,iscan],az[start:end],el[start:end]) *\
                               atmos_coefficient[iband,ichannel,iscan,0]
                        tod[iband,ichannel,:] -= median_filter[iband,:N] * medfilt_coefficient[iband,ichannel,iscan,0]
                        tod[iband,ichannel,:] -= amdl
                        tod[iband,ichannel,:] -= np.nanmedian(tod[iband,ichannel,:])
                
                #tod /= self.calfactors[ifeed,:,:,None] # Calibrate to Jupiter temperature scale
                # Then average together the channels
                wnoise = wnoise_auto[:,:,iscan,0]
                fnoise = fnoise_fits[:,:,iscan,:]
                fnoise_power = fnoise[:,:,0] * np.sqrt(fnoise[:,:,1]**fnoise[:,:,2])

                simple_weights = 1./wnoise**2
                noise_weights = 1./(wnoise**2 + fnoise_power**2)
                noise_weights[(wnoise == 0) | np.isnan(noise_weights) | np.isinf(noise_weights)] = 0
                simple_weights[(wnoise == 0) | np.isnan(simple_weights) | np.isinf(simple_weights)] = 0

                #channels = (self.channelmask[ifeed].flatten() == False)
                #channels = np.where((channels))[0]

                tod    = np.reshape(tod,(tod.shape[0]*tod.shape[1], tod.shape[2]))
                noise_weights = np.reshape(noise_weights,(noise_weights.shape[0]*noise_weights.shape[1]))
                simple_weights = np.reshape(simple_weights,(simple_weights.shape[0]*simple_weights.shape[1]))
                blanks = np.ones(tod.shape)
                noise_weights  = blanks*noise_weights[:,None]
                simple_weights = blanks*simple_weights[:,None]

                simple_weights[np.isnan(tod)] = 0
                noise_weights[np.isnan(tod)] = 0
                tod[np.isnan(tod)] = 0

                
                for ichan, (flow,fhigh) in enumerate(zip(np.arange(8)+26,np.arange(8)+27)):
                    sel = np.where(((self.frequency >= flow) & (self.frequency < fhigh)))[0]
                    top = np.sum(tod[sel,:]*noise_weights[sel,:],axis=0)
                    bot = np.sum(blanks[sel,:]*noise_weights[sel,:],axis=0)
                    self.all_tod[index,ichan,start:end] = top/bot
                    self.all_weights[index,ichan,start:end] = bot
                    self.all_frequency[ichan] = (fhigh+flow)/2.
                self.correlation_matrix[iscan,xfeed.flatten(),yfeed.flatten()]  = stats.correlation(self.all_tod[...,scan_samples]).flatten()



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
            os.chmod(self.outfile,0o664)
            shutil.chown(self.outfile, group=self.permissions_group)

        # Store datasets in root
        dnames = ['tod','weights','cal_factors','frequency']
        dsets = [self.all_tod, self.all_weights,self.cal_factors, self.all_frequency]

        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)

        output.attrs['version'] = __level3_version__
        output['cal_factors'].attrs['source'] = self.cal_source
                        
        output.close()
        
        if self.level3 in data.keys():
            del data[self.level3]
        data[self.level3] = h5py.ExternalLink(self.outfile,'/')

        stats = data['level2/Statistics']
        if 'correlation_matrix' in stats:
            del stats['correlation_matrix']
        dset = stats.create_dataset('correlation_matrix',  data= self.correlation_matrix)
        dset.attrs['level3'] = f'{self.level3}'
        dset.attrs['BW'] = 1
