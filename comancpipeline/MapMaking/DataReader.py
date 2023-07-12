import numpy as np
import h5py
from matplotlib import pyplot
from tqdm import tqdm
import healpy as hp
from comancpipeline.Tools import  Coordinates
from comancpipeline.Tools.median_filter import medfilt

from comancpipeline.MapMaking import MapTypes, OffsetTypes

from scipy.signal import find_peaks

def BinModel(el_in,tod_in):
    elmin = np.min(el_in)
    elmax = np.max(el_in)
    binwidth = 0.1 # degrees
    nbins = int((elmax-elmin)/binwidth)
    eledges = np.linspace(elmin,elmax,nbins+1)
    elmids = (eledges[1:]+eledges[:-1])/2.
    elmodel = np.histogram(el_in,eledges,weights=tod_in)[0]/\
              np.histogram(el_in,eledges)[0]
    return np.interp(el_in, elmids, elmodel)


def GetFeeds(file_feeds, selected_feeds):

    feed_indices = np.array([np.argmin(np.abs(f-file_feeds)) for i,f in enumerate(selected_feeds)])
    distances = np.array([np.abs(f-file_feeds[feed_indices[i]]) > 0 for i, f in enumerate(selected_feeds)])
    gd = (distances == 0)
    return feed_indices[gd]

class ReadDataLevel2:

    def __init__(self,
                 filelist,
                 feeds=1,
                 flag_spikes=False,
                 offset_length=50,
                 ifeature=5,
                 iband=0,
                 ifreq=0,
                 keeptod=False,
                 subtract_sky=False,
                 feed_weights=None,
                 medfilt_stepsize=500,
                 medfilt_name='none',
                 map_info={},
                 **kwargs):

        self.medfilt_stepsize=medfilt_stepsize
        self.medfilt_name=medfilt_name
        # -- constants -- a lot of these are COMAP specific
        self.ifeature = ifeature
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.keeptod = keeptod
        self.iband = int(iband)
        self.ifreq = ifreq
        self.psds = None
        self.psdfreqs = None
        self.feed_weights=feed_weights

        # SETUP MAPS:
        crval = map_info['crval']
        cdelt = map_info['cdelt']
        crpix = map_info['crpix']
        ctype = map_info['ctype']
        nxpix = int(map_info['nxpix'])
        nypix = int(map_info['nypix'])




        # READ PARAMETERS
        self.offset_length = offset_length
        self.flag_spikes = flag_spikes
        self.Feeds  = feeds

        try:
            self.Nfeeds = len(self.Feeds)
            self.Feeds = [int(f) for f in self.Feeds]
        except TypeError:
            self.Feeds = [int(self.Feeds)]
            self.Nfeeds = 1


        #title = parameters['Inputs']['title']
        #self.output_map_filename = f'{title}.fits'
        self.filelist = filelist

        self.naive  = MapTypes.FlatMapType(crval, cdelt,
                                           crpix, ctype,
                                           nxpix=nxpix, nypix=nypix)


        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in tqdm(filelist):
            try:
                self.countDataSize(filename)
            except KeyError:
                print('BAD FILE', filename)

        # Store the Time ordered data as required
        self.pixels = np.zeros(self.Nsamples,dtype=int)
        self.edge_mask = np.zeros(self.Nsamples,dtype=bool)
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
        self.Noffsets  = self.Nsamples//self.offset_length

        # Contains the difference between the TOD and the map average
        self.offset_residuals = OffsetTypes.Offsets(self.offset_length,
                                                    self.Noffsets,
                                                    self.Nsamples)

        for i, filename in enumerate(tqdm(filelist)):
            try:
                self.readPixels(i,filename)
            except KeyError:
                print('BAD FILE', filename)

        #for i, filename in enumerate(tqdm(filelist)):
        #    try:
        #        self.readPSDs(i,filename)
        #    except (KeyError,ValueError):
        #        print('BAD FILE', filename)

        for i, filename in enumerate(tqdm(filelist)):
            #try:
            self.readData(i,filename)
            #except (KeyError,ValueError):
            #    print('BAD FILE', filename)

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
        scan_edges = d['averaged_tod/scan_edges'][:]
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

        dset = d['averaged_tod/tod']
        wei_dset = d['averaged_tod/weights']
        #if 'level2/Flags/sigma_clip_flag' in d:
        #    flags = d['level2/Flags/sigma_clip_flag']
        #else:
        flags = np.zeros((dset.shape[0],dset.shape[-1]),dtype=bool)
        tod_in = np.zeros(dset.shape[-1],dtype=dset.dtype)

        #print
        az = d['spectrometer/pixel_pointing/pixel_az']
        el = d['spectrometer/pixel_pointing/pixel_el']

        scan_edges = d['averaged_tod/scan_edges'][...]
        tod = np.zeros((len(self.FeedIndex), self.datasizes[i]))
        weights = np.zeros((len(self.FeedIndex), self.datasizes[i]))

        # Read in data from each feed
        for index, ifeed in enumerate(self.FeedIndex[:]):

            tod_in = dset[ifeed,self.iband,:]
            wei_in = wei_dset[ifeed,self.iband,:]

            az_in = az[ifeed,:]
            el_in = el[ifeed,:]

            flags_in = flags[ifeed,:]
            if not isinstance(self.feed_weights,type(None)):
                wei_in /= self.feed_weights[index]**2 # bigger feed weights, mean the weights are... lower?
                #print(self.Feeds[index],self.feed_weights[index])


            #if self.flag_spikes:
            #    spikemask = d['level2/Statistics/Spikes/mask'][...]
            #    wei_in[~spikemask] = 0
            #    tod_in[~spikemask] = 0

            wei_in[flags_in > 0.5] = 0

            # then the data for each scan
            last = 0
            for iscan,(start,end) in enumerate(scan_edges):
                N = int((end-start)//self.offset_length * self.offset_length)
                end = start+N
                if (end-start) == 0:
                    continue
                # if medfilt name already in file - retrieve that median filter
                # else create new medfilt and save it
                if self.medfilt_name != 'none':
                    # if 'medfilt' not in d['level3']:
                    #     d['level3'].create_group('medfilt')
                    # if self.medfilt_name in d['level3/medfilt']:
                    #     zfilter = d['level3/medfilt/{}'.format(self.medfilt_name)]
                    # else:
                    zfilter = self.median_filter(tod_in[start:end])
                    # d['level3/medfilt/{}'.format(self.medfilt_name)] = zfilter

                    #pyplot.plot(tod_in[start:end])
                    tod_in[start:end] -= zfilter
                    #pyplot.plot(tod_in[start:end])
                    #pyplot.plot(zfilter)
                    #pyplot.show()

                # Make model of elevation 
                tod_in[start:end] = tod_in[start:end] -\
                                    BinModel(el_in[start:end],tod_in[start:end]) 
                tod_in[start:end] = tod_in[start:end] -\
                                    BinModel(az_in[start:end],tod_in[start:end]) 

                tod[index,last:last+N]  = tod_in[start:end]
                                          
                weights[index,last:last+N] = wei_in[start:end]
                last += N

        return tod, weights

    def median_filter(self, tod):
        """
        """
        if tod.size > 2*self.medfilt_stepsize:
            z = np.concatenate((tod[::-1],tod,tod[::-1]))
            filter_tod = np.array(medfilt.medfilt(z.astype(np.float64),np.int32(self.medfilt_stepsize)))[tod.size:2*tod.size]
        else:
            filter_tod = np.ones(tod.size)*np.nanmedian(tod)

        return filter_tod[:tod.size]


    def readPSDs(self, i, filename):
        """
        Reads PSDs
        """


        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['level1/spectrometer/feeds'][...], self.Feeds)

        # Now accumulate the TOD into the naive map
        tod, weights     = self.getTOD(i,d)
        nFeeds, nSamples = tod.shape


        if isinstance(self.psds,type(None)):
            self.psdfreqs = d['level2/Statistics/freqspectra'][0,0,0,0,:]
            gd = np.isfinite(self.psdfreqs)
            k = np.arange(len(self.psdfreqs))
            self.psdfreqs[~gd] = np.interp(k[~gd],k[gd],self.psdfreqs[gd])
            nfreq = len(self.psdfreqs)
            self.psds = np.zeros((len(self.Feeds), nfreq))
        data_psds = d['level2/Statistics/fnoise_fits'][self.FeedIndex,...]
        for ifeed,feed_num in enumerate(self.Feeds):
            psdfits  = np.nanmean(data_psds[ifeed],axis=(0,1,2))
            self.psds[ifeed,:] += psdfits[0]**2*((self.psdfreqs/10**psdfits[1])**psdfits[2])/len(self.filelist)
        d.close()

    def readPixels(self, i, filename):
        """
        Reads data
        """


        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = d['spectrometer/pixel_pointing/pixel_ra'][self.FeedIndex,:]
        y  = d['spectrometer/pixel_pointing/pixel_dec'][self.FeedIndex,:]
        az = d['spectrometer/pixel_pointing/pixel_az'][self.FeedIndex,:]
        el = d['spectrometer/pixel_pointing/pixel_el'][self.FeedIndex,:]
        dt = 1./50.

        scan_edges = d['averaged_tod/scan_edges'][...]
        pixels = np.zeros((x.shape[0], self.datasizes[i]))
        speed_mask = np.zeros((x.shape[0], self.datasizes[i]),dtype=bool)
        last = 0

        for iscan, (start,end) in enumerate(scan_edges):
            N = int((end-start)//self.offset_length * self.offset_length)
            end = start+N
            xc = x[:,start:end]
            yc = y[:,start:end]
            azc= az[:,start:end]
            elc= el[:,start:end]
            x_veloc = np.gradient(azc[0],dt)*np.cos(np.nanmean(elc)*np.pi/180.)
            y_veloc = np.gradient(elc[0],dt)
            veloc = np.sqrt(x_veloc**2 + y_veloc**2)

            speed_mask[:,last:last+N] = ( np.abs(veloc) > 0.45 ) | (np.abs(veloc) < 0.1) # deg/s
            yshape = yc.shape
            # convert to Galactic
            if 'GLON' in self.naive.wcs.wcs.ctype[0]:
                rot    = hp.rotator.Rotator(coord=['C','G'])
                gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
                xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi

            pixels[:,last:last+N] = np.reshape(self.naive.getFlatPixels(xc.flatten(),yc.flatten()),yshape)
            last += N


        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels.flatten()
        self.edge_mask[self.chunks[i][0]:self.chunks[i][1]] = speed_mask.flatten()
        d.close()

    def readPixelsHealpix(self, i, filename, nside=4096):
        """
        Creates pixels for healpix map 
        """


        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # We store all the pointing information
        x  = d['spectrometer/pixel_pointing/pixel_ra'][self.FeedIndex,:]
        y  = d['spectrometer/pixel_pointing/pixel_dec'][self.FeedIndex,:]
        az = d['spectrometer/pixel_pointing/pixel_az'][self.FeedIndex,:]
        el = d['spectrometer/pixel_pointing/pixel_el'][self.FeedIndex,:]
        dt = 1./50.

        scan_edges = d['averaged_tod/scan_edges'][...]
        pixels = np.zeros((x.shape[0], self.datasizes[i]))
        speed_mask = np.zeros((x.shape[0], self.datasizes[i]),dtype=bool)
        last = 0

        for iscan, (start,end) in enumerate(scan_edges):
            N = int((end-start)//self.offset_length * self.offset_length)
            end = start+N
            xc = x[:,start:end]
            yc = y[:,start:end]
            azc= az[:,start:end]
            elc= el[:,start:end]
            x_veloc = np.gradient(azc[0],dt)*np.cos(np.nanmean(elc)*np.pi/180.)
            y_veloc = np.gradient(elc[0],dt)
            veloc = np.sqrt(x_veloc**2 + y_veloc**2)

            speed_mask[:,last:last+N] = ( np.abs(veloc) > 0.45 ) | (np.abs(veloc) < 0.1) # deg/s
            yshape = yc.shape
            # convert to Galactic
            if 'GLON' in self.naive.wcs.wcs.ctype[0]:
                rot    = hp.rotator.Rotator(coord=['C','G'])
                gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
                xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi

            pixels[:,last:last+N] = hp.ang2pix(nside, (90-yc)*np.pi/180.,xc*np.pi/180).reshape(yshape)
            last += N


        self.pixels[self.chunks[i][0]:self.chunks[i][1]] = pixels.flatten()
        self.edge_mask[self.chunks[i][0]:self.chunks[i][1]] = speed_mask.flatten()
        d.close()


    def readData(self, i, filename):
        """
        Reads data
        """

        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        self.FeedIndex = GetFeeds(d['spectrometer/feeds'][...], self.Feeds)

        # Now accumulate the TOD into the naive map
        tod, weights     = self.getTOD(i,d)
        nFeeds, nSamples = tod.shape

        this_obsid = int(filename.split('/')[-1].split('-')[1])

        # Remove any bad data
        tod     = tod.flatten()
        weights = weights.flatten()
        bad = np.isnan(tod) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        offpix_chunk= self.offset_residuals.offsetpixels[self.chunks[i][0]:self.chunks[i][1]]
        bad_offsets = np.unique(offpix_chunk[bad])
        for bad_offset in bad_offsets:
            tod[offpix_chunk == bad_offset] = 0
            weights[offpix_chunk == bad_offset] = 0

        weights[self.edge_mask[self.chunks[i][0]:self.chunks[i][1]]] = 0 # mask out turn arounds
        # Store TOD
        if self.keeptod:
            self.all_tod[self.chunks[i][0]:self.chunks[i][1]] = tod*1.
        self.all_weights[self.chunks[i][0]:self.chunks[i][1]] = weights



        # Bin data into maps

        self.naive.sum_data(tod,self.pixels[self.chunks[i][0]:self.chunks[i][1]],weights)

        # And then bin the data into the offsets vector
        self.offset_residuals.accumulate(tod,weights,self.chunks[i])
        d.close()

from mpi4py import MPI

class ReadDataLevel2_MADAM:

    def __init__(self, filelist, parameters,nside=8,
                 ifeature=5,iband=0,ifreq=0,
                 keeptod=False,subtract_sky=False,**kwargs):


        # -- constants -- a lot of these are COMAP specific
        self.ifeature = ifeature
        self.chunks = []
        self.datasizes = []
        self.Nsamples = 0
        self.Nhorns = 0
        self.keeptod = keeptod
        self.iband = int(iband)
        self.ifreq = ifreq

        self.nside = nside

        # READ PARAMETERS
        self.offset_length = parameters['Destriper']['offset']

        self.Feeds  = parameters['Inputs']['feeds']

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

        # Will define Nsamples, datasizes[], and chunks[[]]
        for filename in tqdm(filelist):
            self.countDataSize(filename)


        # Store the Time ordered data as required
        self.pixels = np.zeros(self.Nsamples,dtype=int)
        self.edge_mask = np.zeros(self.Nsamples,dtype=bool)
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
        self.Noffsets  = self.Nsamples//self.offset_length

        for i, filename in enumerate(tqdm(filelist)):
            self.readPixels(i,filename)

        for i, filename in enumerate(tqdm(filelist)):
            self.readData(i,filename)


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

        dset     = d['level3/tod']
        wei_dset = d['level3/weights']
        flags    = d['level2/Flags/sigma_clip_flag']
        tod_in   = np.zeros(dset.shape[-1],dtype=dset.dtype)


        scan_edges = d['level2/Statistics/scan_edges'][...]
        tod = np.zeros((len(self.FeedIndex), self.datasizes[i]))
        weights = np.zeros((len(self.FeedIndex), self.datasizes[i]))

        # Read in data from each feed
        for index, ifeed in enumerate(self.FeedIndex[:]):
            tod_in = dset[ifeed,self.iband,:]
            wei_in = wei_dset[ifeed,self.iband,:]
            if not isinstance(self.feed_weights,type(None)):
                wei_in *= self.feed_weights[index]**2 # bigger feed weights, mean the weights are... lower?
                print(self.feeds[index],self.feed_weights[index])

            flags_in = flags[ifeed,:]
            samples = np.arange(tod_in.size)

            peaks, properties = find_peaks(np.abs(tod_in),prominence=1,width=[0,150])
            widths = (properties['right_ips']-properties['left_ips'])*2.

            for peak,width in zip(peaks,widths):
                wei_in[np.abs(samples-peak) < width] = 0

            wei_in[flags_in > 0.5] = 0

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
        x0  = d['level1/spectrometer/pixel_pointing/pixel_ra'][self.FeedIndex,:]
        y0  = d['level1/spectrometer/pixel_pointing/pixel_dec'][self.FeedIndex,:]
        x   = d['level1/spectrometer/pixel_pointing/pixel_az'][self.FeedIndex,:]
        y   = d['level1/spectrometer/pixel_pointing/pixel_el'][self.FeedIndex,:]
        mjd  = d['level1/spectrometer/MJD'][:]
        for i in range(x.shape[0]):
            x[i],y[i] = Coordinates.h2e_full(x[i],y[i],mjd,Coordinates.comap_longitude,Coordinates.comap_latitude)

        scan_edges = d['level2/Statistics/scan_edges'][...]
        pixels = np.zeros((x.shape[0], self.datasizes[i]))
        last = 0
        for iscan, (start,end) in enumerate(scan_edges):
            N = int((end-start)//self.offset_length * self.offset_length)
            end = start+N
            xc = x[:,start:end]
            yc = y[:,start:end]
            pixels[:,last:last+N] = hp.ang2pix(self.nside,(np.pi/2.-yc*np.pi/180.), xc*np.pi/180.,nest=True)
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


        this_obsid = int(filename.split('/')[-1].split('-')[1])

        # Remove any bad data
        tod     = tod.flatten()
        weights = weights.flatten()
        bad = np.isnan(tod) | (self.pixels[self.chunks[i][0]:self.chunks[i][1]] == -1)
        tod[bad] = 0
        weights[bad] = 0

        #bad_offsets = np.unique(offpix_chunk[bad])
        #for bad_offset in bad_offsets:
        #    tod[offpix_chunk == bad_offset] = 0
        #    weights[offpix_chunk == bad_offset] = 0

        # Store TOD
        if self.keeptod:
            self.all_tod[self.chunks[i][0]:self.chunks[i][1]] = tod*1.
        self.all_weights[self.chunks[i][0]:self.chunks[i][1]] = weights
