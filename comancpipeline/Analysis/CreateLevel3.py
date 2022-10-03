# Routines for downsampling the level 2 data to wide band chunks for continuum data analysis
# Data will be cleaned such that it is ready to enter the destriper:
# 1) From Statistics : Remove atmosphere and baselines 
# 2) From astrocalibration : Apply Jupiter calibration to each feed
from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
from scipy.optimize import minimize

from mpl_toolkits.axes_grid1 import make_axes_locatable

#from matplotlib.axes import inset_axes
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

import time
import os

import shutil

from tqdm import tqdm

__level3_version__='v081722'

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
                 cal_database=None,
                 astro_cal=True, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.name = 'CreateLevel3'

        self.output_dirs = output_dirs
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends

        self.database = database + '_{}'.format(os.getpid())
        self.cal_database=cal_database

        self.level2=level2
        self.level3=level3
        self.cal_source=cal_source
        self.nearest_calibrator = 0

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

        #if not 'Statistics' in data[self.level2].keys():
        #    self.logger(f'{fname}:{self.name}:Error: No {self.level2}/Statistics found?')
        #    return data

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

        db = FileTools.safe_hdf5_open(self.cal_database,'r')

        gains = {'Gains':np.zeros((20,8))} #N_feeds, N_bands

        obsids = {k:[] for k in range(19)}
        for obsid, grp in db.items():
            for ifeed in range(19):
                if grp['CalFeedMask'][ifeed]:
                    obsids[ifeed] += [int(obsid)]

        obsids = obsids
        for ifeed in range(19):
            if len(np.array(obsids[ifeed])) == 0:
                continue
            idx = np.argmin(np.abs(np.array(obsids[ifeed])-this_obsid))
            self.nearest_calibrator = str(obsids[ifeed][idx])
            gains['Gains'][ifeed] = db[self.nearest_calibrator]['CalGains'][ifeed,:]

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
            idx = int(feed-1)
            cal_factors = gains['Gains'][idx,...]

        tod = tod/cal_factors[0,...,None]
        weights = weights*cal_factors[0,...,None]**2
        bad = ~np.isfinite(tod) | ~np.isfinite(weights)
        tod[bad] = 0
        weights[bad] = 0


        #channel_mask = self.get_channel_mask(this_obsid,feed)
        #tod[~channel_mask] = 0
        #weights[~channel_mask] = 0

        return tod, weights, cal_factors[0]

    def clean_tod(self,d,ifeed,feed):
        """
        Subtracts the median filter and atmosphere from each channel per scan
        """
        scan_edges = d[f'{self.level2}/Statistics/scan_edges'][...]
        nscans = scan_edges.shape[0]

        feed_tod = d[f'{self.level2}/averaged_tod'][ifeed,:,:,:]
        mask     = np.zeros(feed_tod.shape[-1],dtype=bool)

        N2 = int((feed_tod.shape[-1]//2)*2)
        weights = feed_tod[...,:N2]
        weights = 1./np.nanstd(feed_tod[...,1:N2:2]-feed_tod[...,0:N2:2],axis=-1)**2
        weights = np.repeat(weights[:,None],feed_tod.shape[-1],axis=-1)
        for iscan,(start,end) in enumerate(scan_edges):
            mask[start:end] = True

        new_shape = (feed_tod.shape[0],feed_tod.shape[1],feed_tod.shape[2])
        return np.reshape(feed_tod,new_shape), np.reshape(weights,new_shape), mask

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
        feed_weights[~np.isfinite(feed_tod)] = 0
        feed_tod[~np.isfinite(feed_tod)] = 0
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
        self.all_cal_factors = np.ones((20,4,64))
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

        if not 'pixel_pointing' in output:
            data['level1/spectrometer'].copy('pixel_pointing',output)
            data['level1/spectrometer'].copy('feeds',output)
            data['level1'].copy('comap',output)
            data['level2/Statistics'].copy('scan_edges',output)
            if 'Flags' in data['level2']:
                data['level2'].copy('Flags',output)
            if not 'Flags' in output:
                output.create_group('Flags')
            if 'Spikes' in 'level2/Statistics':
                data['level2/Statistics'].copy('Spikes',output['Flags'])
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


class Level3FnoiseStats(BaseClasses.DataStructure):
    """
    Calculate statistics based on the level 3 data only.
    """

    def __init__(self, allowed_sources = ['co','fg','GField',
                                          'Field','TauA','CasA',
                                          'Jupiter','jupiter','CygA'],
                 nbins=50, 
                 samplerate=20, 
                 database = None,
                 medfilt_stepsize=1000,
                 figure_dir = 'figures/Level3FnoiseStats',
                 make_figures = False,
                 **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.name = 'Level3FnoiseStats'
        self.nbins = int(nbins)
        self.samplerate = samplerate
        self.medfilt_stepsize=medfilt_stepsize
        self.allowed_sources = allowed_sources

        self._figure_dir = figure_dir
        if rank == 0:
            if not os.path.exists(self._figure_dir):
                os.makedirs(self._figure_dir)

        self.database = database + '_{}'.format(os.getpid())
        self.make_figures = make_figures
    def __str__(self):
        return f'Running {self.name}'

    def run(self, data):
        """
        Expects a level2 file structure to be passed.
        """
        fname = data.filename.split('/')[-1]
        # First we need:
        # 1) The TOD data
        # 2) The feature bits to select just the observing period
        # 3) Elevation to remove the atmospheric component
        tod   = data[f'level3/tod'][...]
        az    = data['level3/pixel_pointing/pixel_az'][...]
        el    = data['level3/pixel_pointing/pixel_el'][...]
        feeds = np.arange(1,tod.shape[0]+1,dtype=int)#data['level1/spectrometer/feeds'][:]
        pointing_feeds = data['level3/feeds'][:]
        bands = [b.decode('ascii') for b in data['level1/spectrometer/bands'][:]]
        scan_edges = data['level2/Statistics/scan_edges'][...]

        obsid = self.getObsID(data)

        # Looping over Feed - Band - Channel, perform 1/f noise fit
        nFeeds, nBands, nSamples = tod.shape
        nScans = len(scan_edges)

        self.output_data = {'powerspectra':np.zeros((nFeeds, nBands, nScans, self.nbins)),
                            'fnoise_fits':np.zeros((nFeeds, nBands, nScans, 3)),
                            'atmos_fits':np.zeros((nFeeds, nBands, nScans, 3)),
                            'atmos_errs':np.zeros((nFeeds, nBands, nScans, 3)),
                            'filtered_tod':np.zeros((nFeeds, nBands, tod.shape[-1])),
                            'freqspectra': np.zeros((self.nbins,)),
                            'cov5s': np.zeros((nFeeds, nFeeds, nScans)),
                            '10Hz_inband_C':np.zeros((nFeeds, nFeeds, nScans)),
                            '10Hz_inband_rms':np.zeros((nFeeds, nScans)),
                            '9Hz_lowband_C':np.zeros((nFeeds, nFeeds, nScans)),
                            '9Hz_lowband_rms':np.zeros((nFeeds, nScans))}


        pbar = tqdm(total=(nFeeds*nBands*nScans),desc=self.name)
        for iscan,(start,end) in enumerate(scan_edges):
            for ifeed in range(nFeeds):
                #if ifeed != 0:
                #    continue
                if not feeds[ifeed] in pointing_feeds:
                    continue
                pointing_ifeed = np.argmin((feeds[ifeed] - pointing_feeds)**2)
                if feeds[ifeed] == 20:
                    pbar.update(nBands)
                    continue
                for iband in range(nBands):
                    t = np.arange(tod[ifeed,iband,start:end].size)/self.samplerate
                    scantod = tod[ifeed,iband,start:end]
                    gd = np.isfinite(scantod)
                    scantod[~gd] = np.interp(t[~gd],t[gd],scantod[gd])
                    atmos_filter,atmos,atmos_errs = self.FitAtmosAndGround(scantod,
                                                                           az[pointing_ifeed,start:end],
                                                                           el[pointing_ifeed,start:end])

                    resid = scantod - atmos_filter
                    tod_filter = self.median_filter(resid)
                    resid = resid - tod_filter
                    
                    
                    w_auto = stats.AutoRMS(resid)

                    ps, nu, f_fits, w_auto = self.FitPowerSpectrum(resid,
                                                                   w_auto)

                    self.output_data['powerspectra'][ifeed,iband,iscan] = ps
                    self.output_data['freqspectra'][:] = nu
                    self.output_data['filtered_tod'][ifeed,iband,start:end] = resid
                    self.output_data['atmos_fits'][ifeed,iband,iscan] = atmos
                    self.output_data['atmos_errs'][ifeed,iband,iscan] = atmos_errs
                    self.output_data['fnoise_fits'][ifeed,iband,iscan] = f_fits

                    if self.make_figures:
                        ref_frequency = 1./2.
                        fig = pyplot.figure()
                        ax1 = pyplot.subplot(111)
                        pyplot.plot(nu,ps)
                        pyplot.plot(nu,self.Model_Plot(f_fits,nu,w_auto,ref_frequency))
                        pyplot.axvline(ref_frequency,color='k',ls='--')
                        pyplot.axhline(f_fits[0],color='k',ls='--')
                        pyplot.yscale('log')
                        pyplot.xscale('log')
                        pyplot.xlabel('Frequency (Hz)')
                        pyplot.ylabel('Power (K$^2$)')
                        pyplot.title('Feed {:02d} : Band {:02d} Scan{:02d} '.format(feeds[ifeed],iband,iscan))
                        pyplot.text(0.05,0.25, r'$\sigma_{2s}=$'+'{:.4f} K'.format(np.sqrt(f_fits[0])),
                                    transform=ax1.transAxes,size=15,ha='left')
                        pyplot.text(0.05,0.175,r'$\alpha=$'+'{:.4f}'.format(f_fits[1]),
                                    transform=ax1.transAxes,size=15,ha='left')
                        pyplot.text(0.05,0.10, r'$\sigma_{w}=$'+'{:.4f} K'.format(np.sqrt(f_fits[2])),
                                    transform=ax1.transAxes,size=15,ha='left')


                        #axin1 = ax1.inset_axes([0.7,0.7,0.25,0.25])
                        #im = axin1.imshow(c5)
                        #axin1.set_xticklabels([])
                        #axin1.set_yticklabels([])

                        axin2 = ax1.inset_axes([0.6,0.45,0.4,0.25])
                        im = axin2.plot(self.downsample_tod(resid[None,:],
                                                            int(5*self.samplerate))[0])
                        axin2.set_xticklabels([])
                        axin2.set_yticklabels([])


                        if not os.path.exists(f'{self.figure_dir}/{obsid}'):
                            os.makedirs(f'{self.figure_dir}/{obsid}')
                        pyplot.savefig(f'{self.figure_dir}/{obsid}/Fnoise_Feeds{feeds[ifeed]:02d}_Band{iband:02d}_Scan{iscan:02d}.png')

                        #pyplot.show()
                        pyplot.close(fig)
                    pbar.update(1)

            cs,c5 = self.calculate_covariances(self.output_data['filtered_tod'][:,0,start:end], [1./self.samplerate, 5])
            self.output_data['cov5s'][:,:,iscan] = c5

            limits = [[9.95,10.05],[8.95,9.05]]
            inband,lowband = self.bandpass_signals(self.output_data['filtered_tod'][:,0,start:end],
                                                          limits)
            self.output_data['10Hz_inband_C'][:,:,iscan] = inband['C']
            self.output_data['10Hz_inband_rms'][:,iscan] = inband['rms']
            self.output_data['9Hz_lowband_C'][:,:,iscan] = lowband['C']
            self.output_data['9Hz_lowband_rms'][:,iscan] = lowband['rms']

            if self.make_figures:
                fig = pyplot.figure()
                ax = pyplot.subplot(121)
                im = pyplot.imshow(inband['C'],vmin=0,vmax=1)
                pyplot.title('9.95-10.05Hz',size=10)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='5%', pad=0.05)
                pyplot.colorbar(im,cax=cax)
                ax = pyplot.subplot(122)
                im = pyplot.imshow(lowband['C'],vmin=0,vmax=1)
                pyplot.title('8.95-9.05Hz',size=10)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='5%', pad=0.05)
                pyplot.colorbar(im,cax=cax)
                pyplot.suptitle(f'Spike Feed{feeds[ifeed]:02d} Band{iband:02d} Scan{iscan:02d}')
                pyplot.savefig(f'{self.figure_dir}/{obsid}/Spike10Hz_Feeds{feeds[ifeed]:02d}_Band{iband:02d}_Scan{iscan:02d}.png')
                pyplot.close(fig)
                
                fig = pyplot.figure()
                ax = pyplot.subplot(121)
                im = pyplot.imshow(cs,vmin=0,vmax=1)
                pyplot.title('25Hz',size=10)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='5%', pad=0.05)
                pyplot.colorbar(im,cax=cax)
                
                ax=pyplot.subplot(122)
                im=pyplot.imshow(c5,vmin=0,vmax=1)
                pyplot.title('0.2Hz',size=10)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right',size='5%', pad=0.05)
                pyplot.colorbar(im,cax=cax)
                
                pyplot.suptitle(f'Atmos Feed{feeds[ifeed]:02d} Band{iband:02d} Scan{iscan:02d}')
                pyplot.savefig(f'{self.figure_dir}/{obsid}/Atmos_Feeds{feeds[ifeed]:02d}_Band{iband:02d}_Scan{iscan:02d}.png')
                pyplot.close(fig)


        pbar.close()

    def __call__(self,data):
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')


        self.source = self.getSource(data)
        self.figure_dir = f'{self._figure_dir}/{self.source}'
        if rank == 0:
            if not os.path.exists(self.figure_dir):
                os.makedirs(self.figure_dir)


        comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {self.source} - {comment}')

        if self.checkAllowedSources(data, self.source, self.allowed_sources):
            return data

        if any([s in self.source for s in ['jupiter','CygA','Jupiter','TauA','CasA']]):
            self.isCalibrator = True
        else:
            self.isCalibrator = False 

        if ('Sky nod' in comment) | ('Engineering Test' in comment):
            return data

        if ('level2/Statistics/fnoise_fits' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        #data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Measuring noise stats.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing noise stats to level 3 file ({fname})')
        self.write(data)
        self.write_database(data)

        return data

    def write(self,data):
        """
        Write out fitted statistics to the level 2 file
        """
        fname = data.filename.split('/')[-1]

        lvl3 = data['level3']

        for (dname, dset) in self.output_data.items():
            if dname in lvl3:
                del lvl3[dname]
            lvl3.create_dataset(dname,  data=dset)

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

        if 'level3' in grp:
            lvl3 = grp['level3']
        else:
            lvl3 = grp.create_group('level3')

        for (dname, dset) in self.output_data.items():
            if dname == 'filtered_tod':
                continue
            if dname in lvl3:
                del lvl3[dname]
            lvl3.create_dataset(dname,  data=dset)


        output.close()

    def PowerSpectrum(self, tod):
        """
        Calculates the bin averaged power spectrum
        """
        nu = np.fft.fftfreq(tod.size, d=1/self.samplerate)
        binEdges = np.logspace(np.log10(nu[1]), np.log10(nu[nu.size//2-1]), self.nbins+1)
        ps     = np.abs(np.fft.fft(tod))**2/tod.size

        counts = np.histogram(nu[1:nu.size//2], binEdges)[0]
        signal = np.histogram(nu[1:nu.size//2], binEdges, weights=ps[1:nu.size//2])[0]
        freqs  = np.histogram(nu[1:nu.size//2], binEdges, weights=nu[1:nu.size//2])[0]

        return freqs/counts, signal/counts, counts

    def Model_Plot(self,P,x,rms,ref_frequency):
        return self.Model_rms([np.log10(P[0]),P[1],np.log10(P[2])], x, rms,ref_frequency)
    def Model_rms(self, P, x, rms,ref_frequency):
        return 10**P[0] * (x/ref_frequency)**P[1] + 10**P[2]

    def KneeFrequency(self,P,white_noise,ref_frequency):
        if P[1] != 0:
            return ref_frequency * (white_noise/10**P[0])**(1/P[1])
        else:
            return np.inf

    def Error(self, P, x, y,e, rms,ref_frequency,model):
        error = np.abs(y/e)
        chi = (np.log(y) - np.log(model(P,x,rms,ref_frequency)))/error
        return np.sum(chi**2)

    def FitPowerSpectrum(self, tod, tsys_rms):
        """
        Calculate the power spectrum of the data, fits a 1/f noise curve, returns parameters
        """
        auto_rms = stats.AutoRMS(tod)
        nu, ps, counts = self.PowerSpectrum(tod)

        # Only select non-nan values
        # You may want to increase min counts,
        # as the power spectrum is non-gaussian for small counts
        good = (counts > 50) #& ( (nu < 0.03) | (nu > 0.05)) & np.isfinite(ps)

        ref_frequency = 1./2. # Hz
        ps_nonan = ps[np.isfinite(ps)]
        nu_nonan = nu[np.isfinite(ps)]
        try: # Catch is all the data is bad
            ref = np.argmin((nu_nonan - ref_frequency)**2) 
        except ValueError:
            return ps, nu, [0,0,0], auto_rms
        args = (nu[good], ps[good],auto_rms/np.sqrt(counts[good]), auto_rms, ref_frequency,self.Model_rms)
        bounds =  [[None,None],[-3,0],[None,None]]
        P0 = [np.log10(ps_nonan[ref]),-1,np.log10(auto_rms**2)]

        # We will do an initial guess
        P1 = minimize(self.Error, P0, args= args, bounds = bounds)
        knee = self.KneeFrequency(P1.x, 10**P1.x[2], ref_frequency)

        fits = [10**P1.x[0], P1.x[1], 10**P1.x[2]]


        return ps, nu, fits, auto_rms

    def downsample_tod(self,stod,nbin):
        N = stod.shape[-1]
        nsteps = int(N//nbin)
        N2  = int(nsteps*nbin)
        mtod = np.nanmean(np.reshape(stod[:,:N2],(stod.shape[0], nsteps,nbin)),axis=-1)
        return mtod

    def calculate_covariances(self,stod, nbins):
        out = []
        for nbin_sec in nbins:
            N = stod.shape[1]
            nbin   = int(nbin_sec*self.samplerate)
            nsteps = int(N//nbin)
            N2  = int(nsteps*nbin)
            mtod = np.nanmean(np.reshape(stod[:,:N2],(stod.shape[0], nsteps,nbin)),axis=-1)
            mtod = (mtod - np.nanmean(mtod,axis=-1)[:,None])/np.nanstd(mtod,axis=-1)[:,None]
        
            out += [mtod.dot(mtod.T)/mtod.shape[-1]]
            
        return out

    def bandpass_signals(self,stod, limits,fs=50):
        out = []
        for (low,high) in limits:
            filtered_data = []
            for i in range(stod.shape[0]):
                filtered_data += [butter_bandpass_filter(stod[i],low,high,fs)]
            filtered_data = np.array(filtered_data)
            rms = np.nanstd(filtered_data,axis=-1)
            f = (filtered_data - np.nanmean(filtered_data,axis=-1)[:,None])/\
                rms[:,None]
            C = f.dot(f.T)/f.shape[1]
            out += [{'rms':rms, 'C':C}]
        return out


    def median_filter(self,tod):
        """
        Calculate this AFTER removing the atmosphere.
        """
        if tod.size > 2*self.medfilt_stepsize:
            z = np.concatenate((tod[::-1],tod,tod[::-1]))
            filter_tod = np.array(medfilt.medfilt(z.astype(np.float64),np.int32(self.medfilt_stepsize)))[tod.size:2*tod.size]
        else:
            filter_tod = np.ones(tod.size)*np.nanmedian(tod)

        return filter_tod[:tod.size]

    def FitAtmosAndGround(self,_tod,_az,_el,mask=None,niter=100):
        if isinstance(mask,type(None)):
            mask = np.ones(_tod.size,dtype=bool)

        tod =_tod[mask]
        az  =_az[mask]
        el  =_el[mask]
        # Fit gradients
        dlength = tod.size

        templates = np.ones((3,tod.size))
        templates[0,:] = az
        if np.abs(np.max(az)-np.min(az)) > 180:
            high = templates[0,:] > 180
            templates[0,high] -= 360
        templates[0,:] -= np.median(templates[0,:])
        templates[1,:] = 1./np.sin(el*np.pi/180.)

        a_all = np.zeros((niter,templates.shape[0]))

        for a_iter in range(niter):
            sel = np.random.uniform(low=0,high=dlength,size=dlength).astype(int)

            cov = np.sum(templates[:,None,sel] * templates[None,:,sel],axis=-1)
            z = np.sum(templates[:,sel]*tod[None,sel],axis=1)
            try:
                a_all[a_iter,:] = np.linalg.solve(cov, z).flatten()
            except:
                a_all[a_iter,:] = np.nan

        fits,errs =  np.nanmedian(a_all,axis=0),stats.MAD(a_all,axis=0)
        tod_filter = np.sum(templates[:,:]*fits[:,None],axis=0)

        # interpolate to mask
        tod_filter_all = np.zeros(_tod.size)
        tod_filter_all[mask] = tod_filter
        t = np.arange(_tod.size)
        tod_filter_all[~mask] = np.interp(t[~mask],t[mask],tod_filter)
        
        return tod_filter_all, fits, errs


    def RemoveAtmosphere(self, tod, el):
        """
        Remove 1/sin(E) relationship from TOD
        """
        A = 1/np.sin(el*np.pi/180) # Airmass
        pmdl = np.poly1d(np.polyfit(A, tod,1))
        return tod- pmdl(A), pmdl
