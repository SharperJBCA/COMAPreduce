import concurrent.futures

import numpy as np
from comancpipeline.Analysis.BaseClasses import DataStructure
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import WCS, Coordinates, Filtering, Fitting, Types, ffuncs, binFuncs, stats, CaliModels
from scipy.optimize import fmin, leastsq, minimize
from scipy.interpolate import interp1d
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter,maximum_filter

from matplotlib import pyplot
import glob
from astropy.time import Time
from datetime import datetime

from comancpipeline.Tools import WCS
from comancpipeline.Tools.WCS import DefineWCS
from comancpipeline.Tools.WCS import ang2pix
from comancpipeline.Tools.WCS import ang2pixWCS
from statsmodels import robust
from tqdm import tqdm

from functools import partial
import copy 
from comancpipeline.Tools.median_filter import medfilt

import h5py

import emcee

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import os
import shutil

from scipy.stats import binned_statistic

__version__ = 'v1'

sourcecodes = {'jupiter':0,
               'TauA':1,
               'CasA':2,
               'CygA':3}

class AstroCal(DataStructure):
    """
    Read in all the calibration factors, update Level 2 files with astro factors

    Warning: Only run in the second pipeline run


    Init - construct the calibation table
    Call - Interpolate calibration factors to file observation time
    """

    def __init__(self, feeds='all',
                 prefix='AstroCal',
                 astro_cal_dir = '',
                 astro_cal_version='v3',
                 allowed_sources= ['jupiter','TauA','CasA','CygA','mars'],
                 level2='level2',**kwargs):
        """
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)
        self.name = 'AstroCal'
        self.feeds_select = feeds
        
        self.prefix = prefix
        self.allowed_sources = allowed_sources

        self.level2=level2

        self.astro_cal_version='v3'
        self.astro_cal_dir = astro_cal_dir
        self.calibration_table_file = f'{astro_cal_dir}/GainTable.hd5'

        self.models = {'jupiter':CaliModels.JupiterFlux,
                       'TauA': CaliModels.TauAFlux,
                       'CygA': CaliModels.CygAFlux,
                       'CasA': CaliModels.CasAFlux}

        self.construct_calibration_table()

    def build_gains(self):
        """
        """
        astrocal_files = []
        for fname in tqdm(glob.glob(f'{self.astro_cal_dir}/{self.prefix}*.hd5')):
            try:
                h = h5py.File(fname,'r')
            except OSError:
                continue

            if not 'SourceFittingVersion' in h.attrs:
                h.close()
                continue
            if (h.attrs['SourceFittingVersion'] == self.astro_cal_version):
                astrocal_files += [fname]
            h.close()

        astrocal_files = astrocal_files
        counts = {}
        for fname in tqdm(astrocal_files):
            try:
                data = h5py.File(fname,'r')
            except OSError:
                continue
            try:
                source = self.get_source(data)
            except OSError:
                data.close()
                continue

            if not source in counts:
                counts[source] = 0
            counts[source] += 1
            data.close()
        # Now build up calibration table
        nFeed, nBand = 20,8
        gains = {source:{'time':np.empty((counts[source]),dtype=object),
                         'mjd':np.zeros((counts[source])),
                         'frequency':np.zeros((counts[source],nBand)),
                         'obsid':np.zeros((counts[source]),dtype=int),
                         'sourcecode':np.zeros((counts[source]),dtype=int),
                         'gain':np.zeros((counts[source],nFeed,nBand)),
                         'error':np.zeros((counts[source],nFeed,nBand))} for source in counts.keys()}


        counts = {source:0 for source in counts.keys()}
        print(gains.keys())
        print(counts.keys())

        for fname in tqdm(astrocal_files):
            obsid = int(fname.split('-')[1])
            try:

                data = h5py.File(fname,'r')
            except OSError:
                continue
            try:
                fluxes, nu = self.get_flux(data)
                mjd,time    = self.get_mjd(data)
                source = self.get_source(data)
            except OSError:
                data.close()
                continue

            if not source in counts:
                counts[source] = 0
            try:
                feeds = self.get_feeds(data)
            except OSError:
                data.close()
                continue

            mdl_flux = self.models[source](nu,mjd,return_jansky=True,allpos=True)
            gain  = fluxes['Values']/mdl_flux
            error = fluxes['Errors']/mdl_flux

            gains[source]['obsid'][counts[source]] = obsid
            if source in sourcecodes:
                gains[source]['sourcecode'][counts[source]] = sourcecodes[source]
            else:
                gains[source]['sourcecode'][counts[source]] = -1

            gains[source]['time'][counts[source]] = time
            gains[source]['mjd'][counts[source]] = mjd
            gains[source]['frequency'][counts[source],:] = nu
            gains[source]['gain'][counts[source],feeds,:] = gain 
            gains[source]['error'][counts[source],feeds,:] = error
            counts[source] += 1
            data.close()
        return gains

    def write_gain_table(self,times, gains, frequency,obsid,codes):
        """
        
        """
        if os.path.exists(self.calibration_table_file):
            os.remove(self.calibration_table_file)

        output = h5py.File(self.calibration_table_file,'a')
        # Store datasets in root
        dnames = ['MJD','gains','frequency','obsid','sourcecode']
        dsets  = [times, gains, frequency,obsid,codes]
                  
        for (dname, dset) in zip(dnames, dsets):
            if dname in output:
                del output[dname]
            output.create_dataset(dname,  data=dset)
        output.close()

    def construct_calibration_table(self):
        """
        0) If calibration table exists and no overwrite - read and skip
        1) Find all astro calibration files.
        2) Transform each source measurement into Gain factors
        3*) <Correct for atmospheric absorption if not using vane>
        4) Store all gain factors per day into a file
        """

        #if os.path.exists(self.calibration_table_file) & (not self.overwrite):
        ##    self.table_data = h5py.File(self.calibration_table_file,'r')
        #    return 

        # Get files in correct format
        testfile = 'test_source_data.npy'
        if not os.path.exists(testfile):
            gains = self.build_gains()
            np.save(testfile,[gains])
        else:
            gains = np.load(testfile,allow_pickle=True).flatten()[0]


        # Flatten Gains into days
        allgains = np.concatenate([v['gain'] for k,v in gains.items()])
        alltimes = np.concatenate([v['mjd'] for k,v in gains.items()])
        allfrequency = np.concatenate([v['frequency'] for k,v in gains.items()])
        allobsid = np.concatenate([v['obsid'] for k,v in gains.items()])
        allcodes = np.concatenate([v['sourcecode'] for k,v in gains.items()])

        print(allfrequency.shape)
        # Remove bad fits
        nObs, nFeed, nBand = allgains.shape
        meds = np.zeros((nFeed,nBand))
        mads = np.zeros((nFeed,nBand))
        scale = 3
        pyplot.figure(figsize=(20,20))
        for ifeed in range(nFeed):
            for iband in range(nBand):
                meds[ifeed,iband] = np.nanmedian(allgains[:,ifeed,iband])
                mads[ifeed,iband] = stats.MAD(allgains[:,ifeed,iband])
                bad = np.abs(allgains[:,ifeed,iband] - meds[ifeed,iband]) > mads[ifeed,iband]*scale
                allgains[bad,ifeed,iband] = np.nan

            if ifeed < 5:
                ax1=pyplot.subplot(2,2,1)
                pyplot.plot(allfrequency[0,:],meds[ifeed,:],'.',label=(ifeed+1))
            elif (ifeed >= 5) & (ifeed < 10):
                ax2=pyplot.subplot(2,2,2)
                pyplot.plot(allfrequency[0,:],meds[ifeed,:],'.',label=(ifeed+1))
            elif (ifeed >= 10) & (ifeed < 15):
                ax3=pyplot.subplot(2,2,3)
                pyplot.plot(allfrequency[0,:],meds[ifeed,:],'.',label=(ifeed+1))
            elif ifeed >= 15:
                ax4=pyplot.subplot(2,2,4)
                pyplot.plot(allfrequency[0,:],meds[ifeed,:],'.',label=(ifeed+1))

        for ax in [ax1,ax2,ax3,ax4]:
            pyplot.sca(ax)
            pyplot.legend(loc='upper left',prop={'size':10})
            pyplot.ylim(0.6,0.9)
            pyplot.grid()
            pyplot.xlabel('Frequency (GHz)')
            pyplot.ylabel('Gain')
        pyplot.savefig('figures/AstroCal/average_gains.png')
        pyplot.clf()

        # Write a table to AstroCal directory
        self.write_gain_table(alltimes,allgains,allfrequency,allobsid,allcodes)

    #def plots(self):
        nFeed = 20
        nBand = 8
        colors = ['C1','C2','C3','C4']
        for ifeed in range(nFeed):
            fig = pyplot.figure(figsize=(20,12))
            for iband in range(nBand):
                pyplot.subplot(2,4,1+iband)
                data = []
                labels=[]
                for i,(k,v) in enumerate(gains.items()):
                    data += [v['gain'][:,ifeed,iband]]
                    labels +=[k]
                pyplot.hist(data,30,range=[0.5,1.5], density=True, stacked=True,label=labels,color=colors)
                data= np.concatenate(data)
                pyplot.axvline(np.nanmedian(data),color='k',ls='--',lw=2)
                pyplot.legend()
            pyplot.suptitle(f'Feed: {ifeed+1}')
            pyplot.tight_layout()
            pyplot.savefig(f'figures/AstroCal/{ifeed+1:02d}_histogram.png')        
            pyplot.close(fig)


        fig = pyplot.figure()
        for ifeed in range(nFeed):
            fig = pyplot.figure(figsize=(20,12))
            for iband in range(nBand):
                pyplot.subplot(2,4,1+iband)

                for i,(k,v) in enumerate(gains.items()):
                    pyplot.errorbar(v['time'],v['gain'][:,ifeed,iband],fmt='.',capsize=3,color=colors[i], yerr=v['error'][:,ifeed,iband],label=k)
                pyplot.legend()
                pyplot.ylim(0.5,1.5)
                fig.autofmt_xdate()
                pyplot.title(iband)
                pyplot.grid()
            pyplot.suptitle('Feed: {ifeed+1}')
            pyplot.savefig(f'figures/AstroCal/{ifeed+1:02d}_timeline.png')        
            pyplot.close(fig)
            
    def get_mjd(self,data):
        """
        """
        fname = data.filename.split('/')[-1]
        date = fname.split('-')[2:6]
        date[-1] = date[-1].split('_')[0]
        date = ':'.join(date)
        time = Time(datetime.strptime(date,'%Y:%m:%d:%H%M%S'))

        return time.mjd,time.datetime

    def get_source(self,data):
        """
        """
        fname = data.filename.split('/')[-1]
        fdir  = data.filename.split(fname)[0]
        fname = fname.split(f'{self.prefix}_')[-1]
        lvl2 = h5py.File(f'{fdir}/../{fname}','r')
        
        source = self.getSource(lvl2)
        lvl2.close()
        return source

    def get_feeds(self,data):
                                 
        fname = data.filename.split('/')[-1]
        fdir  = data.filename.split(fname)[0]
        fname = fname.split(f'{self.prefix}_')[-1]
        lvl2 = h5py.File(f'{fdir}/../{fname}','r')
        
        feeds, feedlist, feeddict = self.getFeeds(lvl2,'all')
                 
        lvl2.close()
        feeds = (feeds-1).astype(int)
        return feeds

    def get_flux(self,data):
        """
        Convert values into flux density
        """
        kb = 1.38064852e-23
        c  = 2.99792458e8
        scale = 2 * kb * (1e9/ c)**2 * 1e26
        out = {'Values':None, 'Errors':None}
        for k in out.keys():
            amps = data[f'Gauss_{k}']['Amp'][...]
            sigx = data[f'Gauss_{k}']['sigx'][...]
            sigy = data[f'Gauss_{k}']['sigy'][...]
            freq = data[f'Gauss_Values']['frequency'][...]

            flux = 2*np.pi*amps*sigx*sigy*(np.pi/180.)**2 * scale*freq[None,...]**2
            out[k] = np.reshape(flux,(flux.shape[0],flux.shape[1]*flux.shape[2]))

        return out, freq.flatten()
            

    def __str__(self):
        return 'Calculating calibration tables'

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        fdir  = data.filename.split(fname)[0]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        comment = self.getComment(data)
        self.source = self.getSource(data)

        if self.checkAllowedSources(data, self.source, self.allowed_sources):
            return data

        if 'Sky nod' in comment:
            return data
        if 'test' in comment:
            return data

        if isinstance(self.output_dir, type(None)):
            self.output_dir = f'{fdir}/{self.prefix}'

        outfile = '{}/{}_{}'.format(self.output_dir,self.prefix,fname)
        if os.path.exists(outfile) & (not self.overwrite):
            self.logger(f'{fname}:{self.name}: Source fits for {fname} already exist ({outfile}).')
            return data 


        data = self.setReadWrite(data)


        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing source fits to {outfile}.')
        if not self.nodata:
            self.write(data)
        self.logger(f'{fname}:{self.name}: Done.')

        return data
