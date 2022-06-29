# Here we have a number of statistics that are applied to the level 3 data.
# We are doing this because after averaging we want to check the noise,
# spikes, etc... are all correctly being spotted and flagged.
# 
import h5py
from mpi4py import MPI
import numpy as np
import time
from astropy.time import Time
from matplotlib import pyplot
from datetime import datetime
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting

from comancpipeline.Tools import Coordinates, Types, stats, FileTools
from comancpipeline.Tools.median_filter import medfilt
from scipy.signal import find_peaks
from scipy.optimize import minimize

class Level3Statistics(BaseClasses.DataStructure):

    def __init__(self,level3='level3',database=None,**kwargs):
        """
        """
        super().__init__(**kwargs)
        self.name = 'Level3Statistics'
        self.level3=level3
        self.database=database
        self.samplerate=50. # Hz
        self.nbins = 20

    def __str__(self):
        return "Calculating level 3 statistics"

    def __call__(self,data):
        """
        Expects a level2 file structure containing level 3 data.
        """
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')

        self.source  = self.getSource(data)
        self.comment = self.getComment(data)

        self.logger(f'{fname}:{self.name}: {self.source} - {self.comment}')

        if ('level3' in data) & (not self.overwrite):
            return data

        # Want to ensure the data file is read/write
        data = self.setReadWrite(data)

        self.logger(f'{fname}:{self.name}: Running statistics.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing ({fname})')
        self.write(data)
        if not isinstance(self.database,type(None)):
            self.write_database(data)
        self.logger(f'{fname}:{self.name}: Done.')

    def run(self, data):
        """
        """
        # First look for spikes
        self.spike_mask = self.fit_spikes(data)

        # Next get noise statistics
        fnoise = self.fit_noise(data)

        self.outputs = {**spike_mask, **fnoise}
        
    def AperPhot(self,nu, ps, x0, width):
        """
        """
        
        xlo_0 = x0 - width
        xlo_1 = x0 - width/2.

        xhi_0 = x0 + width/2.
        xhi_1 = x0 + width

        annu = ((nu > xlo_0) & (nu < xlo_1)) | ((nu > xhi_0) & (nu < xhi_1))
        aper = (nu > x0 - width/2.) & (nu < x0 + width/2.)
        Nannu = len(ps[annu])
        Naper = len(ps[aper])
        flux_10Hz = np.sum(ps[aper]) - np.nanmedian(ps[annu])*Naper
        errs_10Hz = np.nanstd(ps[annu])*np.sqrt(Naper)

        return flux_10Hz, errs_10Hz, annu, aper

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

        # 
        binEdges = np.logspace(np.log10(nu[1]), np.log10(nu[nu.size//2-1]), nu.size//4 + 1)
        counts_lo = np.histogram(nu[1:nu.size//2], binEdges)[0]
        signal_lo = np.histogram(nu[1:nu.size//2], binEdges, weights=ps[1:nu.size//2])[0]
        freqs_lo  = np.histogram(nu[1:nu.size//2], binEdges, weights=nu[1:nu.size//2])[0]

        ps_lo = signal_lo/counts_lo
        nu_lo = freqs_lo/counts_lo

        flux_10Hz, errs_10Hz,an0,ap0 = self. AperPhot(nu_lo,ps_lo,10, 0.1)
        #flux_9Hz, errs_9Hz,an1,ap1 = self. AperPhot(nu_lo,ps_lo,9, 0.1)

        #pyplot.subplot(2,1,1)
        #pyplot.plot(tod)
        #pyplot.subplot(2,1,2)
        #pyplot.plot(nu_lo,ps_lo)
        #pyplot.xscale('log')
        #pyplot.yscale('log')
        #pyplot.show()

        return freqs/counts, signal/counts, counts, flux_10Hz, errs_10Hz

    def Model_rms(self, P, x,ref_frequency):
        return 10**P[0] * (x/ref_frequency)**P[1] + 10**P[2]

    def Error(self, P, x, y,e,ref_frequency,model):
        error = np.abs(y/e)
        chi = (np.log(y) - np.log(model(P,x,ref_frequency)))/error
        return np.sum(chi**2)

    def FitPowerSpectrum(self, tod):
        """
        Calculate the power spectrum of the data, fits a 1/f noise curve, returns parameters
        """
        auto_rms = stats.AutoRMS(tod)
        nu, ps, counts, flux_10Hz, errs_10Hz = self.PowerSpectrum(tod)

        # Only select non-nan values
        # You may want to increase min counts,
        # as the power spectrum is non-gaussian for small counts
        good = (counts > 1) #& ( (nu < 0.03) | (nu > 0.05)) & np.isfinite(ps)

        ref_frequency = 2. # Hz
        ps_nonan = ps[np.isfinite(ps)]
        nu_nonan = nu[np.isfinite(ps)]
        try: # Catch is all the data is bad
            ref = np.argmin((nu_nonan - ref_frequency)**2) 
        except ValueError:
            return ps, nu, [0,0,0], auto_rms
        args = (nu[good], ps[good],auto_rms/np.sqrt(counts[good]), ref_frequency,self.Model_rms)
        bounds =  [[None,None],[-3,0],[None,None]]
        P0 = [np.log10(ps_nonan[ref]),-1,np.log10(auto_rms**2)]

        # We will do an initial guess
        P1 = minimize(self.Error, P0, args= args, bounds = bounds)

        return nu, ps,P1.x, auto_rms, flux_10Hz, errs_10Hz


    def fit_noise(self,data):
        """
        Fit the 1/f noise in the data
        """
        tod = data[f'{self.level3}/tod'][...]
        nbands = tod.shape[1]

        fnoise = {'fits': np.zeros((len(self.feeds), nbands, 3)),
                  'auto': np.zeros((len(self.feeds), nbands, 1)),
                  '10Hz': np.zeros((len(self.feeds), nbands, 2))}
        self.feeds, self.feedidx,_ = self.getFeeds(data,'all')
        for ifeed, feed in enumerate(self.feeds):
            for iband in range(nbands):
                gd = np.isfinite(tod[ifeed,iband]) & ~self.spike_mask['spike_mask'][ifeed,iband]
                v,ps, fits, auto_rms,flux_10Hz, errs_10Hz = self.FitPowerSpectrum(tod[ifeed,iband,gd])

                fnoise['fits'][ifeed,iband] = fits
                fnoise['auto_rms'][ifeed,iband] = auto_rms
                fnoise['10Hz'][ifeed,iband] = flux_10Hz, errs_10Hz

        return fnoise

    def fit_spikes(self,data):
        """
        Fit transient spikes in the data just using a simple sigma clipping
        """

        self.feeds, self.feedidx,_ = self.getFeeds(data,'all')
        scale = 10

        tod = data[f'{self.level3}/tod'][...]

        N2 = (tod.shape[-1]//2) * 2
        auto_rms = np.nanstd(tod[:,:,:N2:2]-tod[:,:,1:N2:2],axis=-1)/np.sqrt(2)

        mask = np.zeros((tod.shape[0],tod.shape[2]),dtype=bool)
        iband = -2
        for ifeed,feed in enumerate(self.feeds):
            peaks = np.where((np.abs(tod[ifeed,iband]) > auto_rms[ifeed,iband]*scale))[0]
            peaks = np.concatenate((peaks, peaks-1, peaks+1))
            peaks = peaks[(peaks >= 0) & (peaks < tod.shape[2])]
            mask[ifeed,peaks] = True

        return {'spike_mask':mask}

    def write(self,data):
        """
        Write out fitted statistics to the level 3 file
        """
        fname = data.filename.split('/')[-1]
        if not self.level3 in data:
            return
        lvl3 = data[self.level3]
        if not 'Statistics' in lvl3:
            statistics = lvl3.create_group('Statistics')
        else:
            statistics = lvl3['Statistics']

        for (dname, dset) in self.outputs.items():
            if dname in statistics:
                del statistics[dname]
            statistics.create_dataset(dname,  data=dset)

    @staticmethod
    def write_database(data,database,dataout):
        """
        Write out the statistics to a common statistics database for easy access
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

        if 'Level3Stats' in grp:
            del grp['Level3Stats']
        stats = grp.create_group('Level3Stats')

        for dname, dset in dataout.items():
            if dname in stats:
                del stats[dname]
            stats.create_dataset(dname,  data=dset)
        output.close()
