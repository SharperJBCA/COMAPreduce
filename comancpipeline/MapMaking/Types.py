import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
from comancpipeline.Tools import  binFuncs, stats

from comancpipeline.Analysis import Statistics
from scipy import signal

import time
import os

def butter_highpass(cutoff, fs, order=5,btype='highpass'):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq 
    b,a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    return b,a

def butt_bandpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order,btype='bandpass')
    y = signal.filtfilt(b,a,data)
    return data-y

def butt_highpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order)
    y = signal.filtfilt(b,a,data)
    return y
def butt_lowpass(data, cutoff, fs, order=3):
    b,a = butter_highpass(cutoff,fs,order=order,btype='lowpass')
    y = signal.filtfilt(b,a,data)
    return y



def removeplane(img, slce=0.4):
    """
    Remove a quadratic 2D plane from an image
    """
    img[img == 0] = np.nan

    xr, yr = np.arange(slce*img.shape[0],(1-slce)*img.shape[0],dtype=int),\
             np.arange(slce*img.shape[1],(1-slce)*img.shape[1],dtype=int)
    x, y = np.meshgrid(xr,yr)

    
    subimg = img[xr[0]:xr[-1]+1,yr[0]:yr[-1]+1]
    imgf = subimg[np.isfinite(subimg)].flatten()

    vecs = np.ones((5,imgf.size))
    vecs[0,:] = x[np.isfinite(subimg)].flatten()
    vecs[1,:] = y[np.isfinite(subimg)].flatten()
    vecs[2,:] = x[np.isfinite(subimg)].flatten()**2
    vecs[3,:] = y[np.isfinite(subimg)].flatten()**2

    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    img -= (xv[0]*x    + xv[1]*y    + \
            xv[2]*x**2 + xv[3]*y**2 + \
            xv[4])
    return img

class Data:
    """
    Reads in the TOD, stores it immediate in a naive map
    """
    def __init__(self,filelist, parameters, frequency=0, band=0):
        
        self.minimum_scanlength = 2000

    def GetFeeds(self, feedlist, feeds):
        """
        Return feed index position
        """
        output = []
        for feed in feeds:
            pixel = np.where((feedlist == feed))[0]
            if (len(pixel) == 1):
                output += [pixel]

        output = np.array(output).flatten().astype(int)
        return output



    def GetScanPositions(self,d):
        """
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,
        provides indices for the positions of scans in masked array
        """

        # make it so that you have a gap, only use data where the telescope is moving

        elcurrent = np.abs(d['level1/hk/antenna0/driveNode/elDacOutput'][:])
        elutc = d['level1/hk/antenna0/driveNode/utc'][:]
        mjd = d['level1/spectrometer/MJD'][:]
        select = np.where((elcurrent > np.max(elcurrent)*0.8))[0] # these are when the telescope is changing position
        #if len(select) == 0:


        dselect = select[1:]-select[:-1]
        ends = np.concatenate((np.where((dselect > 10))[0], np.array([len(dselect)-1])))

        indices = []
        for e in select[ends]:
            end_idx = np.argmin((elutc[e]-mjd)**2)
            indices += [end_idx]

        mean_scan_time = np.mean(elutc[ends[1:]] - elutc[ends[:-1]])*24*3600
        step = mean_scan_time*0.1
        mask = np.zeros(len(mjd)).astype(bool)
        samples = np.arange(len(mjd))

        buffer_size = 50
        buffer_multiply = 5
        for iend, end in enumerate(indices):
            if (iend ==0):
                mask[samples < (end+step*buffer_size)] = True
            elif (iend == (len(indices)-1)):
                mask[samples > (end-step*buffer_size)] = True
            else:
                mask[(samples > (end - buffer_multiply*buffer_size)) & (samples < (end + buffer_multiply*buffer_size))] = True


        # Map indices
        oldindex = np.arange(len(mjd))[~mask] # old positions
        newindex = np.arange(len(oldindex)) # new positions
        mapOld2New = {o:n for (o,n) in zip(oldindex,newindex)}
        mapNew2Old = {n:o for (o,n) in zip(oldindex,newindex)}

        diff_mask = mask[1:].astype(int) - mask[:-1].astype(int)
        mask_select = np.where((diff_mask > 0))[0] # old positions
        end_indices = np.unique([0] + [mapOld2New[i] for i in mask_select] )

        if end_indices[-1] > newindex.size:
            end_indices[-1] = newindex.size-1
            

        starts, ends = end_indices[:-1],end_indices[1:] 

        start_final = []
        end_final = []
        for start,end in zip(starts,ends):
            i0,i1 = mapNew2Old[start], mapNew2Old[end]
            if (end-start) < self.minimum_scanlength:
                mask[i0:i1] = True
            else:
                start_final += [start]
                end_final += [end]

        return (mask == False), np.array(start_final).astype(int), np.array(end_final).astype(int)


    def featureBits(self,features, target):
        """
        Return list of features encoded into feature bit
        """
        # Select Features
        features[features == 0] = 0.1
        p2 = np.floor(np.log(features)/np.log(2))
        
        select = (p2 != 13) & (p2 != -1)
        a = np.where(select)[0]
        select[a[:1000]] = False
        return select

    def selectData(self, features, target, d):
        """ calls both GetScanPositions and featuresBits"""

        scan_mask, self.scan_starts, self.scan_ends = self.GetScanPositions(d)

        selectFeature = self.featureBits(features.astype(float), self.ifeature)

        self.select_mask = (scan_mask & selectFeature)
        
        return self.select_mask

    def countDataSize(self,filename):
        """
        Get size of data for this file
        """
        
        d = h5py.File(filename,'r')
        features = d['spectrometer/features'][:]
        select = self.selectData(features.astype(float), self.ifeature, d)
        N = len(features[select])
        d.close()

        N = (N//self.offsetLen) * self.offsetLen

        N = N*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N/self.Nfeeds)]
        self.Nsamples += int(N)

    def processChunks(self,tod, step=10000):
        
        from scipy.linalg import inv
        from scipy.signal import medfilt
        nSteps = tod.shape[1]//step
        templates = np.ones((step,2))

        #nu = np.fft.fftfreq(tod.shape[1], d=1/50)
        #ps1 = np.abs(np.fft.fft(tod[0,:]))**2
        for i in range(nSteps):
            lo = i*step
            if i < (nSteps - 1):
                hi = (i+1)*step
            else:
                hi = tod.shape[1]
                templates = np.ones((hi - lo,2))

            #templates[:,0] = medfilt(np.median(tod[:,lo:hi],axis=0),151)
            # 
            
            #print(mdl.shape, tod[:,lo:hi].shape)
            
            medfilt_tod = np.zeros((tod.shape[0],hi-lo))
            binDown = 5
            binCount = int((hi-lo)/binDown +0.5)
            if binCount*binDown < tod.shape[1]:
                binCount += 1
            binEdges = np.linspace(0,hi-lo, binCount+1)
            positions = np.arange(hi-lo)
            w = np.histogram(positions, binEdges)[0]
            for feed in range(tod.shape[0]):
               s = np.histogram(positions, binEdges, weights=tod[feed,lo:hi])[0]
               medfilt_tod[feed,:] = np.repeat(s/w, binDown)[:hi-lo]
            tod[:,lo:hi] -= medfilt_tod #mdl
        return tod

    def readData(self, i, filename):
        """
        Reads data
        """

        gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/CalVanes/'
        gfile = filename.split('/')[-1].split('.')[0]+'_TsysGainRMS.pkl'
        gainDF = pd.read_pickle(gdir+gfile)
        idx = pd.IndexSlice
        gains = gainDF.loc(axis=0)[idx[:,:,'Gain',self.Feeds,self.band]].values.astype('float')
        gains = np.nanmedian(gains,axis=1)

        d = h5py.File(filename,'r')

        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]
        

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = (d['spectrometer/pixel_pointing/pixel_ra'][...])[Feeds,selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['spectrometer/pixel_pointing/pixel_dec'][...])[Feeds,selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()

        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el  = el[...,0:self.datasizes[i]]


        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > self.naive.npix] = -1
        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels
        
        # Now accumulate the TOD into the naive map
        tod = ((d['spectrometer/band_average'][Feeds,:,:])[:,self.band,:])[:,selectFeature]
        tod = tod[...,0:self.datasizes[i]]

        #tod = np.zeros((todin.shape[0], todin.shape[1], np.sum(selectFeature)))
        #print(tod.shape)
        # print(features.shape)
        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        for j in range(tod.shape[0]):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])
            tod[j,:] /= gains[j]

            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2
            #print('Horn', j, rms)

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        #pyplot.scatter(np.arange(tod[0,0,:].size),tod[0,0,:], c=np.log(features)/np.log(2))
        #pyplot.show()
        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.

        
        self.naive[(band,frequency)].accumulate(tod,weights,pixels)
        self.hits[(band,frequency)].accumulatehits(pixels)

    def offsetResidual(self, i, filename):
        """
        Reads data
        """
        gdir = '/scratch/nas_comap1/sharper/COMAP/runcomapreduce/AncillaryData/CalVanes/'
        gfile = filename.split('/')[-1].split('.')[0]+'_TsysGainRMS.pkl'
        gainDF = pd.read_pickle(gdir+gfile)
        idx = pd.IndexSlice
        gains = gainDF.loc(axis=0)[idx[:,:,'Gain',self.Feeds,self.band]].values.astype('float')
        gains = np.nanmedian(gains,axis=1)


        d = h5py.File(filename,'r')

        # -- Only want to look at the observation data
        features = d['spectrometer/features'][:]
        selectFeature = self.featureBits(features.astype(float), self.ifeature)
        features = features[selectFeature]

        # --- Feed position indices can change
        Feeds = self.GetFeeds(d['spectrometer/feeds'][...], self.Feeds)


       
        # Now accumulate the TOD into the naive map
        el  = (d['spectrometer/pixel_pointing/pixel_el'][...])[Feeds,selectFeature]
        el = el[...,0:self.datasizes[i]]
        tod = (d['spectrometer/band_average'][Feeds,:,:])[:,self.band,selectFeature]
        tod = tod[...,0:self.datasizes[i]]
        t = np.arange(tod.shape[-1])
        weights = np.ones(tod.shape)
        for j in range(tod.shape[0]):
            bad = np.isnan(tod[j,:])
            tod[j,bad] = np.interp(t[bad], t[~bad], tod[j,~bad])
            pmdl = np.poly1d(np.polyfit(1./np.sin(el[j,:]*np.pi/180.), tod[j,:],1))
            tod[j,:] -= pmdl(1./np.sin(el[j,:]*np.pi/180.))
            tod[j,:] -= np.nanmedian(tod[j,:])
            tod[j,:] /= gains[j]

            N = tod.shape[0]//2 * 2
            rms = np.nanstd(tod[j,1:N:2] - tod[j,0:N:2])
            weights[j,:] *= 1./rms**2

        weights = weights.flatten()
        tod = tod.flatten()
        bad = (np.isnan(tod)) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        self.residual.accumulate(tod,weights,self.naive.output,self.pixels,self.chunks[i])

    def skyPixels(self,i, d,Feeds, selectFeature):
        """
        Returns the pixel coordinates in the WCS frame
        """

        # We store all the pointing information
        x  = (d['level1/spectrometer/pixel_pointing/pixel_ra'][...])[Feeds[:,None],selectFeature]
        x  = x[...,0:self.datasizes[i]].flatten()
        y  = (d['level1/spectrometer/pixel_pointing/pixel_dec'][...])[Feeds[:,None],selectFeature]
        y  = y[...,0:self.datasizes[i]].flatten()


        el  = (d['level1/spectrometer/pixel_pointing/pixel_el'][...])[Feeds[:,None],selectFeature]
        el  = el[...,0:self.datasizes[i]]


        pixels = self.getFlatPixels(x,y)
        pixels[pixels < 0] = -1
        pixels[pixels > self.naive.npix] = -1

        return pixels


class DataLevel2AverageHPX(Data):

    def __init__(self, *args,nside=4096,medfilt_stepsize=1500,keeptod=False,subtract_sky=False,**kwargs):
        
        super().__init__(*args,**kwargs)
        filelist, parameters = args
        
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = 5
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.Nbands = 4
        self.keeptod = keeptod
        self.medfilt_stepsize = medfilt_stepsize

        self.nfeeds_all = 18
        self.nfreqs_all = 64

        # READ PARAMETERS
        self.offsetLen = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1


        # READ ANY ANCILLARY DATA: MASKS/CALIBRATION FACTORS
        if 'channel_mask' in parameters['Inputs']:
            self.channelmask = np.load(parameters['Inputs']['channel_mask'],allow_pickle=True).astype(bool)
        else:
            self.channelmask = np.zeros((self.nfeeds_all, self.Nbands, self.nfreqs_all)).astype(bool)

        if 'gain_mask' in parameters['Inputs']:
            self.gainmask = np.load(parameters['Inputs']['gain_mask'],allow_pickle=True).astype(bool)
        else:
            self.gainmask = np.zeros((self.nfeeds_all, self.Nbands, self.nfreqs_all)).astype(bool)

        self.channelmask = self.channelmask | self.gainmask

        # Read in calibration factors
        if 'feed_calibration_factors' in parameters['Inputs']:
            self.calfactors = np.load(parameters['Inputs']['feed_calibration_factors'])
        else:
            self.calfactors = np.ones((self.nfeeds_all, self.Nbands, self.nfreqs_all))

        
        # FILTER OUT CHANNELS OUTSIDE OF FREQUENCY BOUNDS
        # Frequency Range - need to make this more dynamic
        frequencies = np.array((np.arange(self.nfreqs_all,0,-1)-1,
                                np.arange(self.nfreqs_all) + self.nfreqs_all,
                                np.arange(self.nfreqs_all,0,-1)-1 + 128, 
                                np.arange(self.nfreqs_all)+192)) + 0.5
        frequencies = frequencies*32./1024. + 26.


        upperFreqBound = parameters['Inputs']['upper_frequency']
        lowerFreqBound = parameters['Inputs']['lower_frequency']
        title = parameters['Inputs']['title']
        bounds =  (frequencies > upperFreqBound) | (frequencies < lowerFreqBound) 
        self.output_map_filename = '{}_{}-{}.fits'.format(title,int(upperFreqBound),int(lowerFreqBound))

        self.subtract_sky = subtract_sky
        if self.subtract_sky:
            self.model_sky = hp.read_map(self.output_map_filename)
            self.model_sky[np.isnan(self.model_sky) | (self.model_sky == hp.UNSEEN)] = 0

        for ifeed in range(self.channelmask.shape[0]):
            self.channelmask[ifeed,...] = self.channelmask[ifeed,...] | np.reshape(bounds, self.channelmask[ifeed].shape)
        

        # SETUP MAPS:
        self.nside = nside 
        self.naive  = ProxyHealpixMap(self.nside)
        self.hits   = ProxyHealpixMap(self.nside)


        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in tqdm(filelist):
            self.countDataSize(filename)
        self.pixels = np.zeros(self.Nsamples,dtype=int)

        # If we want to keep all the TOD samples for plotting purposes...
        if self.keeptod:
            self.todall = np.zeros(self.Nsamples)
        self.allweights = np.zeros(self.Nsamples)


        # First read in all the data
        # Remember we want to solve Ax = b,
        # "b" contains all the data, so we construct that now:
        # 1a) Create a naive binned map
        # 1b) Sum all the data into offsets
        # 2) Subtract the naive weighted map from the offsets
        # "b" residual vector is saved in residual Offset object
        Noffsets  = self.Nsamples//self.offsetLen
        self.residual = Offsets(self.offsetLen, Noffsets, self.Nsamples)

        for i, filename in enumerate(tqdm(filelist)):
            self.readPixels(i,filename)      

        # Removing Blank pixels
        self.naive.remove_blank_pixels(self.pixels)
        self.hits.remove_blank_pixels(self.pixels)
        self.pixels = self.naive.modify_pixels(self.pixels)

        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)        
        self.naive.average()
        self.residual.accumulate(-self.naive.output[self.pixels],self.allweights,[0,self.pixels.size])
        self.residual.average()


    def countDataSize(self,filename):
        """
        Opens each datafile and determines the number of samples

        Uses the features to select the correct chunk of data
        """
        
        try:
            d = h5py.File(filename,'r')
        except:
            print(filename)
            return 

        N = 0
        scan_edges = d['level2/Statistics/scan_edges'][:]
        for (start,end) in scan_edges:
            N += (end-start)//self.offsetLen * self.offsetLen
        d.close()

        N = N*self.Nfeeds

        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        self.datasizes += [int(N/self.Nfeeds)]
        self.Nsamples += int(N)


    def skyPixelsHPX(self,i, d,feedindex):
        """
        Returns the pixel coordinates in the WCS frame
        """

        # We store all the pointing information
        x  = d['level1/spectrometer/pixel_pointing/pixel_ra'][feedindex,:][:,self.select_mask]
        x  = x[:,0:self.datasizes[i]].flatten()
        y  = d['level1/spectrometer/pixel_pointing/pixel_dec'][feedindex,:][:,self.select_mask]
        y  = y[:,0:self.datasizes[i]].flatten()
            
        # convert to Galactic
        rot = hp.rotator.Rotator(coord=['C','G'])
        gb, gl = rot((90-y)*np.pi/180., x*np.pi/180.)

        pixels = hp.ang2pix(self.nside, gb, gl)
        return pixels

    def getTOD(self,i,d):
        """
        Want to select each feed and average the data over some frequency range
        """
        output_filename = 'Output_Fits/{}'.format( d.filename.split('/')[-1])
        if os.path.exists(output_filename):
            os.remove(output_filename)

        tod_shape = d['level2/averaged_tod'].shape
        dset = d['level2/averaged_tod']
        tod_in = np.zeros((tod_shape[1],tod_shape[2],tod_shape[3]),dtype=dset.dtype)
        az = np.zeros((tod_shape[3]),dtype=dset.dtype)
        el = np.zeros((tod_shape[3]),dtype=dset.dtype)

        feeds = d['level1/spectrometer/feeds'][:]
        scan_edges = d['level2/Statistics/scan_edges'][...]

        todall = np.zeros((len(self.FeedIndex), self.datasizes[i])) 
        weights = np.zeros((len(self.FeedIndex), self.datasizes[i])) 

        # Read in data from each feed
        for index, ifeed in enumerate(self.FeedIndex[:]):

            dset.read_direct(tod_in,np.s_[ifeed:ifeed+1,:,:,:])
            d['level1/spectrometer/pixel_pointing/pixel_az'].read_direct(az,np.s_[ifeed:ifeed+1,:])
            d['level1/spectrometer/pixel_pointing/pixel_el'].read_direct(el,np.s_[ifeed:ifeed+1,:])

            # Statistics for this feed
            medfilt_coefficient = d['level2/Statistics/filter_coefficients'][ifeed,...]
            atmos = d['level2/Statistics/atmos'][ifeed,...]
            atmos_coefficient = d['level2/Statistics/atmos_coefficients'][ifeed,...]
            wnoise_auto = d['level2/Statistics/wnoise_auto'][ifeed,...]

            # then the data for each scan
            last = 0
            for iscan,(start,end) in enumerate(scan_edges):
                median_filter = d['level2/Statistics/FilterTod_Scan{:02d}'.format(iscan)][ifeed,...]
                N = int((end-start)//self.offsetLen * self.offsetLen)
                end = start+N
                tod = tod_in[...,start:end]

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


                top = np.sum(tod[channels,:]/wnoise[channels,:]**2,axis=0)
                bot = np.sum(1/wnoise[channels,:]**2)

                todall[index,last:last+N] = top/bot
                weights[index,last:last+N] = bot
                last += N

        return todall, weights


    def readPixels(self, i, filename):
        """
        Reads data
        """    

        
        d = h5py.File(filename,'r')
            
        # --- Feed position indices can change
        self.FeedIndex = self.GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = d['level1/spectrometer/pixel_pointing/pixel_ra'][self.FeedIndex,:]
        y  = d['level1/spectrometer/pixel_pointing/pixel_dec'][self.FeedIndex,:]

        scan_edges = d['level2/Statistics/scan_edges'][...]
        pixels = np.zeros((x.shape[0], self.datasizes[i]))
        last = 0
        for iscan, (start,end) in enumerate(scan_edges):
            N = int((end-start)//self.offsetLen * self.offsetLen)
            end = start+N
            ra  = x[:,start:end]
            dec = y[:,start:end]

            # convert to Galactic
            rot = hp.rotator.Rotator(coord=['C','G'])
            gb, gl = rot((90-dec.flatten())*np.pi/180., ra.flatten()*np.pi/180.)

            pixels[:,last:last+N] = np.reshape(hp.ang2pix(self.nside, gb, gl),ra.shape)
            last += N


        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels.flatten()


    def readData(self, i, filename):
        """
        Reads data
        """    

        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = self.GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)
        
        # Now accumulate the TOD into the naive map
        tod, weights= self.getTOD(i,d)
        nFeeds, nSamples = tod.shape
        
        
        # Remove any bad data
        tod = tod.flatten()
        weights = weights.flatten()
        bad = np.isnan(tod) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        # Store TOD
        if self.keeptod:
            self.todall[self.chunks[i][0]:self.chunks[i][1]] = tod*1.
        self.allweights[self.chunks[i][0]:self.chunks[i][1]] = weights

        
        # Bin data into maps
        self.naive.accumulate(tod,weights,self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.hits.accumulatehits(self.pixels[self.chunks[i][0]:self.chunks[i][1]])
        self.residual.accumulate(tod,weights,self.chunks[i])


class Map:
    """
    Stores pixel information
    """
    def __init__(self,nxpix, nypix,wcs,storehits=False):

        self.storehits = storehits
        # --- Need to create several arrays:
        # 1) Main output map array
        # 2) Signal*Weights array
        # 3) Weights array
        # 4) Hits

        self.wcs = wcs
        self.npix = nypix*nxpix
        self.nypix = nypix
        self.nxpix = nxpix
        self.output = np.zeros(self.npix)
        self.sigwei = np.zeros(self.npix)
        self.wei    = np.zeros(self.npix)
        if self.storehits:
            self.hits = np.zeros(self.npix)

    def clearmaps(self):
        self.output *= 0
        self.sigwei *= 0
        self.wei *= 0
        if self.storehits:
            self.hits *= 0

    def accumulate(self,tod,weights,pixels):
        """
        Add more data to the naive map
        """
        binFuncs.binValues(self.sigwei, pixels, weights=tod*weights)
        binFuncs.binValues(self.wei   , pixels, weights=weights    )
        if self.storehits:
            binFuncs.binValues(self.hits, pixels,mask=weights)

    def accumulatehits(self,pixels):
        binFuncs.binValues(self.sigwei,pixels)

    def binOffsets(self,offsets,weights,offsetpixels,pixels):
        """
        Add more data to the naive map
        """
        binFuncs.binValues2Map(self.sigwei, pixels, offsets*weights, offsetpixels)
        binFuncs.binValues2Map(self.wei   , pixels, weights        , offsetpixels)



    def __call__(self, average=False, returnsum=False):
        if average:
            self.average()
        
        if returnsum:
            return np.reshape(self.sigwei, (self.nypix, self.nxpix))

        return np.reshape(self.output, (self.nypix, self.nxpix))


    def __getitem__(self,pixels, average=False):
        if average:
            self.average()
        return self.output[pixels]

    def average(self):
        self.goodpix = np.where((self.wei != 0 ))[0]
        self.output[self.goodpix] = self.sigwei[self.goodpix]/self.wei[self.goodpix]
    def weights(self):
        return np.reshape(self.wei, (self.nypix, self.nxpix))

class HealpixMap(Map):
    """
    Stores pixel information
    """
    def __init__(self,npix,storehits=False):

        self.storehits = storehits
        # --- Need to create several arrays:
        # 1) Main output map array
        # 2) Signal*Weights array
        # 3) Weights array
        # 4) Hits

        self.npix = npix
        self.output = np.zeros(self.npix)
        self.sigwei = np.zeros(self.npix)
        self.wei    = np.zeros(self.npix)
        if self.storehits:
            self.hits = np.zeros(self.npix)

    def __call__(self):
        self.average()
        return self.output

    def weights(self):
        return self.wei

class ProxyHealpixMap(Map):
    """
    Stores pixel information
    """
    def __init__(self,nside=None, npix=None,storehits=False):

        self.storehits = storehits
        # --- Need to create several arrays:
        # 1) Main output map array
        # 2) Signal*Weights array
        # 3) Weights array
        # 4) Hits

        if isinstance(nside, type(None)):
            self.nside = int(np.sqrt(npix/12.))
        else:
            self.nside= nside
        
        if isinstance(npix, type(None)):
            self.npix = 12*self.nside**2
        else:
            self.npix = npix
        self.output = np.zeros(self.npix)
        self.sigwei = np.zeros(self.npix)
        self.wei    = np.zeros(self.npix)
        if self.storehits:
            self.hits = np.zeros(self.npix)

    def return_hpx_map(self):
        self.average()
        m = np.zeros(12*self.nside**2)
        m[self.uni2pix] = self.output
        return m

    def return_hpx_hits(self):
        m = np.zeros(12*self.nside**2)
        m[self.uni2pix] = self.sigwei
        return m

    def return_hpx_variance(self):
        m = np.zeros(12*self.nside**2)
        m[self.uni2pix] = 1./self.wei
        return m

    def remove_blank_pixels(self,pixels,non_zero=None):
        """
        Remove all the blank pixels so we don't carry around extra memory
        """
        self.uni2pix= np.unique(pixels).astype(int)
        self.pix2uni = {u:k for k,u in enumerate(self.uni2pix)}

        gb, gl = hp.pix2ang(self.nside, self.uni2pix)

        self.npix = self.uni2pix.size
        if isinstance(non_zero,type(None)):
            non_zero = np.where(self.wei != 0)[0]

        self.output = self.output[self.uni2pix]
        self.sigwei = self.sigwei[self.uni2pix]
        self.wei    = self.wei[self.uni2pix]

        print('SIZE CHECK', self.wei.size, self.npix)
        

    def modify_pixels(self,pixels):
        return np.array([self.pix2uni[p] for p in pixels])

    def __call__(self):
        return self.return_hpx_map()

    def weights(self):
        return self.wei


class Offsets:
    """
    Stores offset information
    """
    def __init__(self,offset, Noffsets, Nsamples):
        """
        """
        
        self.Noffsets= int(Noffsets)
        self.offset = int(offset)
        self.Nsamples = int(Nsamples )

        self.offsets = np.zeros(self.Noffsets)

        self.sigwei = np.zeros(self.Noffsets)
        self.wei    = np.zeros(self.Noffsets)

        self.offsetpixels = np.arange(self.Nsamples)//self.offset

    def __getitem__(self,i):
        """
        """
        
        return self.offsets[i//self.offset]

    def __call__(self):
        return np.repeat(self.offsets, self.offset)[:self.Nsamples]


    def clear(self):
        self.offsets *= 0
        self.sigwei *= 0
        self.wei *= 0

    def accumulate(self,tod,weights,chunk):
        """
        Add more data to residual offset
        """
        binFuncs.binValues(self.sigwei, self.offsetpixels[chunk[0]:chunk[1]], weights=tod*weights )
        binFuncs.binValues(self.wei   , self.offsetpixels[chunk[0]:chunk[1]], weights=weights    )


    def average(self):
        self.goodpix = np.where((self.wei != 0 ))[0]
        self.offsets[self.goodpix] = self.sigwei[self.goodpix]/self.wei[self.goodpix]
