import numpy as np
from matplotlib import pyplot
import h5py
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats
from comancpipeline.Tools.median_filter import medfilt
from scipy.ndimage import gaussian_filter1d

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

from tqdm import tqdm

class SigmaClip(DataStructure): 
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, sigma_clip_value=10, medfilt_stepsize=5000):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        self.medfilt_stepsize = int(medfilt_stepsize)
        self.sigma_clip_value = sigma_clip_value

    def __str__(self):
        return "Sigma clipping at {}x the noise".format(self.sigma_clip_value)

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod   = np.nanmean(data['level1/spectrometer/band_average'][...],axis=1)
        az    = data['level1/spectrometer/pixel_pointing/pixel_az'][...]
        el    = data['level1/spectrometer/pixel_pointing/pixel_el'][...]
        feeds = data['level1/spectrometer/feeds'][:]
        nFeeds, nSamples = tod.shape
        scan_edges = data['level2/Statistics/scan_edges'][...]
        self.flags = np.zeros(tod.shape).astype(bool)

        for ifeed in range(nFeeds):
            for (s,e) in scan_edges:
                rtod = tod[ifeed,s:e] - self.median_filter(tod[ifeed,s:e],self.medfilt_stepsize)
                rms = stats.AutoRMS(rtod)
                f = gaussian_filter1d((rtod > rms*self.sigma_clip_value).astype(np.float),50)
                f /= np.max(f)
                self.flags[ifeed,s:e] = (f > 0.5)


    def write(self,data):
        """
        Write out the averaged TOD to a Level2 continuum file with an external link to the original level 1 data
        """        

        if not 'level2' in data:
            return 
        lvl2 = data['level2']
        if not 'Flags' in lvl2:
            flags_grp = lvl2.create_group('Flags')
        else:
            flags_grp = lvl2['Flags']

        dnames = ['sigma_clip_flag']
        dsets  = [self.flags]
        for (dname, dset) in zip(dnames, dsets):
            if dname in flags_grp:
                del flags_grp[dname]            
            flags_grp.create_dataset(dname,  data=dset)

        flags_grp['sigma_clip_flag'].attrs['sigma_clip_value'] = self.sigma_clip_value

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'

        source = data['level1/comap'].attrs['source']
        if not isinstance(source,str):
            source = source.decode('utf-8')
        comment = data['level1/comap'].attrs['comment']
        if not isinstance(comment, str):
            comment = comment.decode('utf-8')
        print(source,comment)

        if 'Sky nod' in comment:
            return data
        # Want to ensure the data file is read/write
        if not data.mode == 'r+':
            filename = data.filename
            data.close()
            data = h5py.File(filename,'r+')

        try:
            self.run(data)
            self.write(data)
        except KeyError:
            pass

        return data        

    def median_filter(self,tod,medfilt_stepsize):
        """
        Calculate this AFTER removing the atmosphere.
        """
        filter_tod = np.array(medfilt.medfilt(tod.astype(np.float64),np.int32(medfilt_stepsize)))
        
        return filter_tod[:tod.size]



