import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools import  binFuncs, stats

from comancpipeline.MapMaking import MapTypes, OffsetTypes

def GetFeeds(file_feeds, selected_feeds):

    feed_indices = np.array([np.argmin(np.abs(f-file_feeds)) for i,f in enumerate(selected_feeds) ])

    return feed_indices

class ReadDataLevel2:

    def __init__(self, filelist, parameters, 
                 ifeature=5,iband=0,ifreq=0,
                 keeptod=False,subtract_sky=False,**kwargs):
        
        
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = ifeature
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.keeptod = keeptod
        self.iband = iband
        self.ifreq = ifreq

        # READ PARAMETERS
        self.offset_length = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']
        print(self.Feeds)
        try:
            self.Nfeeds = len(parameters['Inputs']['feeds'])
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1


        title = parameters['Inputs']['title']
        self.output_map_filename = f'{title}.fits'        


        # SETUP MAPS:
        crval = parameters['Destriper']['crval']
        cdelt = parameters['Destriper']['cdelt']
        crpix = parameters['Destriper']['crpix']
        ctype = parameters['Destriper']['ctype']
        nxpix = int(parameters['Destriper']['nxpix'])
        nypix = int(parameters['Destriper']['nypix'])

        self.naive  = MapTypes.FlatMapType(crval, cdelt, crpix, ctype,nxpix,nypix)


        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in tqdm(filelist):
            self.countDataSize(filename)


        # Store the Time ordered data as required
        self.pixels = np.zeros(self.Nsamples,dtype=int)
        self.all_weights = np.zeros(self.Nsamples)
        if self.keeptod:
            self.all_tod = np.zeros(self.Nsamples)


        # First read in all the data
        # Remember we want to solve Ax = b,
        # "b" contains all the data, so we construct that now:
        # 1a) Create a naive binned map
        # 1b) Sum all the data into offsets
        # 2) Subtract the naive weighted map from the offsets
        # "b" residual vector is saved in residual Offset object
        Noffsets  = self.Nsamples//self.offset_length

        # Contains the difference between the TOD and the map average
        self.offset_residuals = OffsetTypes.Offsets(self.offset_length, Noffsets, self.Nsamples)

        for i, filename in enumerate(tqdm(filelist)):
            self.readPixels(i,filename)      

        
        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)     

        #pyplot.subplot(projection=self.naive.wcs)
        #m = self.naive.get_map()
        #m[m==0]=np.nan
        #pyplot.imshow(m,aspect='auto')
        #pyplot.show()
        
        self.naive.average()

        self.offset_residuals.accumulate(-self.naive.sky_map[self.pixels],self.all_weights,[0,self.pixels.size])
        self.offset_residuals.average()

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
            N += (end-start)//self.offset_length * self.offset_length
        d.close()

        N = N*self.Nfeeds

        # Store the beginning and end point of each file
        self.chunks += [[int(self.Nsamples), int(self.Nsamples+N)]]
        
        # We also want to know how big each file is per feed
        self.datasizes += [int(N/self.Nfeeds)]

        # Finally, add to the total number of files
        self.Nsamples += int(N)


    def getTOD(self,i,d):
        """
        Want to select each feed and average the data over some frequency range
        """

        dset = d['level3/tod']
        wei_dset = d['level3/weights']
        flags = d['level2/Flags/sigma_clip_flag']
        tod_in = np.zeros(dset.shape[-1],dtype=dset.dtype)

        scan_edges = d['level2/Statistics/scan_edges'][...]
        tod = np.zeros((len(self.FeedIndex), self.datasizes[i])) 
        weights = np.zeros((len(self.FeedIndex), self.datasizes[i])) 

        # Read in data from each feed
        for index, ifeed in enumerate(self.FeedIndex[:]):
            #dset.read_direct(tod_in,np.s_[ifeed:ifeed+1,self.iband,:])
            tod_in = dset[ifeed,self.iband,:]
            wei_in = wei_dset[ifeed,self.iband,:]
            flags_in = flags[ifeed,:]
            wei_in[flags_in > 0] = 0

            # then the data for each scan
            last = 0
            for iscan,(start,end) in enumerate(scan_edges):
                N = int((end-start)//self.offset_length * self.offset_length)
                end = start+N
                tod[index,last:last+N]  = tod_in[start:end]
                weights[index,last:last+N] = wei_in[start:end]
                last += N

        return tod, weights


    def readPixels(self, i, filename):
        """
        Reads data
        """    

        
        d = h5py.File(filename,'r')
            
        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = d['level1/spectrometer/pixel_pointing/pixel_ra'][self.FeedIndex,:]
        y  = d['level1/spectrometer/pixel_pointing/pixel_dec'][self.FeedIndex,:]

        scan_edges = d['level2/Statistics/scan_edges'][...]
        pixels = np.zeros((x.shape[0], self.datasizes[i]))
        last = 0
        for iscan, (start,end) in enumerate(scan_edges):
            N = int((end-start)//self.offset_length * self.offset_length)
            end = start+N
            xc = x[:,start:end]
            yc = y[:,start:end]

            # convert to Galactic
            if 'GLON' in self.naive.wcs.wcs.ctype[0]:
                rot    = hp.rotator.Rotator(coord=['C','G'])
                gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
                xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi

            pixels[:,last:last+N] = np.reshape(self.naive.getFlatPixels(xc,yc),yc.shape)
            last += N


        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels.flatten()


    def readData(self, i, filename):
        """
        Reads data
        """    

        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)
        
        # Now accumulate the TOD into the naive map
        tod, weights     = self.getTOD(i,d)
        nFeeds, nSamples = tod.shape
        
        # Remove any bad data
        tod     = tod.flatten()
        weights = weights.flatten()
        bad = np.isnan(tod) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        # Store TOD
        if self.keeptod:
            self.all_tod[self.chunks[i][0]:self.chunks[i][1]] = tod*1.
        self.all_weights[self.chunks[i][0]:self.chunks[i][1]] = weights

        # Bin data into maps
        self.naive.sum_data(tod,self.pixels[self.chunks[i][0]:self.chunks[i][1]],weights)

        # And then bin the data into the offsets vector
        self.offset_residuals.accumulate(tod,weights,self.chunks[i])
