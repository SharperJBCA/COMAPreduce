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

from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting
from comancpipeline.Analysis import Statistics
from scipy import signal

import time
import os

from tqdm import tqdm

__version__='v1'

class CreateLevel3(DataStructure):
    def __init__(self,channel_mask=None, gain_mask=None, calibration_factors=None, **kwargs):
        """
        """
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

    def __str__(self):
        return "Creating Level 3"

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        comment = data['level1/comap'].attrs['comment']
        if not isinstance(comment,str):
            comment = comment.decode('utf-8')

        if 'Sky nod' in comment:
            return data

        if not 'level2' in data.keys():
            print(f'No level 2 data found in {data.filename}')
            return data

        if not 'Statistics' in data['level2'].keys():
            print(f'No level 2 statistics found in {data.filename}')
            return data
        if not 'scan_edges' in data['level2/Statistics'].keys():
            return data
        self.run(data)
        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        self.write(data)
        return data


    def run(self, d):
        """
        Expects a level2 file structure to be passed.
        """
        tod_shape = d['level2/averaged_tod'].shape

        scan_edges = d['level2/Statistics/scan_edges'][...]
        nchannels = 8
        self.all_tod     = np.zeros((tod_shape[0], nchannels, tod_shape[-1])) 
        self.all_weights = np.zeros((tod_shape[0], nchannels, tod_shape[-1])) 
        frequency = d['level1/spectrometer/frequency'][...]
        frequency = np.mean(np.reshape(frequency,(frequency.shape[0],frequency.shape[1]//16,16)) ,axis=-1).flatten()
        
        # Read in data from each feed
        for index, ifeed in enumerate(range(tod_shape[0])):

            todin = d['level2/averaged_tod'][ifeed,:,:,:]
            az = d['level1/spectrometer/pixel_pointing/pixel_az'][ifeed,:]
            el = d['level1/spectrometer/pixel_pointing/pixel_el'][ifeed,:]

            # Statistics for this feed
            medfilt_coefficient = d['level2/Statistics/filter_coefficients'][ifeed,...]
            atmos = d['level2/Statistics/atmos'][ifeed,...]
            atmos_coefficient = d['level2/Statistics/atmos_coefficients'][ifeed,...]
            wnoise_auto = d['level2/Statistics/wnoise_auto'][ifeed,...]

            # Create gain masks/channel masks/calfactors
            if isinstance(self.gainmask, type(None)):
                self.gainmask = np.zeros((tod_shape[0],tod_shape[1],tod_shape[2])).astype(bool)
            if isinstance(self.channelmask, type(None)):
                self.channelmask = np.zeros((tod_shape[0],tod_shape[1],tod_shape[2])).astype(bool)
            if isinstance(self.calfactors, type(None)):
                self.calfactors = np.ones((tod_shape[0],tod_shape[1],tod_shape[2])).astype(bool)

            self.channelmask = self.channelmask | self.gainmask



            # then the data for each scan
            last = 0
            for iscan,(start,end) in enumerate(scan_edges):
                median_filter = d['level2/Statistics/FilterTod_Scan{:02d}'.format(iscan)][ifeed,...]
                N = int((end-start))
                end = start+N
                tod = todin[...,start:end]

                # Subtract atmospheric fluctuations per channel
                for iband in range(4):
                    for ichannel in range(64):
                        if self.channelmask[ifeed,iband,ichannel] == False:
                            amdl = Statistics.AtmosGroundModel(atmos[iband,iscan],az[start:end],el[start:end]) *\
                                   atmos_coefficient[iband,ichannel,iscan,0]
                            tod[iband,ichannel,:] -= median_filter[iband,:N] * medfilt_coefficient[iband,ichannel,iscan,0]
                            tod[iband,ichannel,:] -= amdl
                            tod[iband,ichannel,:] -= np.nanmedian(tod[iband,ichannel,:])
                tod /= self.calfactors[ifeed,:,:,None] # Calibrate to Jupiter temperature scale
                # Then average together the channels
                wnoise = wnoise_auto[:,:,iscan,:]
                channels = (self.channelmask[ifeed].flatten() == False)
                channels = np.where((channels))[0]

                tod    = np.reshape(tod,(tod.shape[0]*tod.shape[1], tod.shape[2]))
                wnoise = np.reshape(wnoise,(wnoise.shape[0]*wnoise.shape[1], wnoise.shape[2]))

                nancheck = np.sum(tod[channels,:],axis=1)
                channels = channels[np.isfinite(nancheck) & (nancheck != 0)]
                nancheck = np.sum(wnoise[channels,:],axis=1)
                channels = channels[np.isfinite(nancheck) & (nancheck != 0)]

                tod, wnoise = tod[channels,:], wnoise[channels,:]
                freq = frequency[channels]

                for ichan, (flow,fhigh) in enumerate(zip([26,27,28,29,30,31,32],[28,29,30,31,32,33,34])):
                    sel = np.where(((freq >= flow) & (freq < fhigh)))[0]
                    sel = sel[10:-10]
                    top = np.sum(tod[sel,:]/wnoise[sel,:]**2,axis=0)
                    bot = np.sum(1/wnoise[sel,:]**2,axis=0)

                    self.all_tod[index,ichan,start:end] = top/bot
                    self.all_weights[index,ichan,start:end] = bot

    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """        

        if not 'level3' in data:
            lvl3= data.create_group('level3')
        else:
            lvl3 = data['level3']

        dnames = ['tod','weights']
        dsets = [self.all_tod, self.all_weights]
        for (dname, dset) in zip(dnames, dsets):
            if dname in lvl3:
                del lvl3[dname]            
            lvl3.create_dataset(dname,  data=dset)
