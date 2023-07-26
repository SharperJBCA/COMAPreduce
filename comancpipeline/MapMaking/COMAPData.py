"""
COMAPData.py -- Read the level 3 files and return the pointing, weights and tod for destriping 
"""
import numpy as np
import h5py
from tqdm import tqdm
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

@dataclass
class COMAPData(object):

    filename : str = None # Filename of the data
    obsid : int = -1 
    offset_length : int = 100 

    @staticmethod
    def get_size(filename, feeds=[0], offset_length=100):
        """ Get the size of the data """
        nfeeds = len(feeds)
        h = h5py.File(filename,'r')
        nsize = int(h['averaged_tod/tod'].shape[-1]//offset_length*offset_length)
        hdu.close()
        return nsize*nfeeds

    def __post_init__(self):
        """ Load the data """
        self.extra_data = {} 

        self.hdu = fits.open(self.filename,memmap=False)

        self.load_data() 
        self.load_weights() 
        self.flag_data() 
        self.calc_parallactic_angle() 
        self.hdu.close() 

    def __del__(self):
        self.hdu.close() 
        try:
            self.clear_memory() 
        except AttributeError:
            pass

    def clear_memory(self):
        del self.I1 
        del self.I2 
        del self.Q1
        del self.Q2
        del self.U1
        del self.U2
        del self.wI1
        del self.wI2
        del self.wQ1
        del self.wQ2
        del self.wU1
        del self.wU2
        del self.dayflag
        del self.sundist

    @property
    def nsize(self):
        if hasattr(self,'I1'):
            return int((self.I1.size//self.offset_length)*self.offset_length)
        else:
            return int((self.hdu[1].data['I1'].size//self.offset_length)*self.offset_length)


    def load_data(self):
        self.I1 = self.hdu[1].data['I1'][:self.nsize]
        self.I2 = self.hdu[1].data['I2'][:self.nsize]
        self.Q1 = self.hdu[1].data['Q1'][:self.nsize]
        self.Q2 = self.hdu[1].data['Q2'][:self.nsize]
        self.U1 = self.hdu[1].data['U1'][:self.nsize]
        self.U2 = self.hdu[1].data['U2'][:self.nsize]
        self.flag = self.hdu[1].data['FLAG'][:self.nsize]
        self.dayflag = self.hdu[1].data['DAYFLAG'][:self.nsize]
        self.sundist = self.hdu[1].data['SUNDIST'][:self.nsize]
        self.mjd = self.hdu[1].data['MJD'][:self.nsize]
        self.az = self.hdu[1].data['AZ'][:self.nsize]
        self.el = self.hdu[1].data['EL'][:self.nsize]
        self.ra = self.hdu[1].data['RA'][:self.nsize]
        self.dec = self.hdu[1].data['DEC'][:self.nsize]

    def load_weights(self):
        self.wI1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['I1_sigma'][0]**2 
        self.wI2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['I2_sigma'][0]**2
        self.wQ1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['Q1_sigma'][0]**2
        self.wQ2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['Q2_sigma'][0]**2
        self.wU1 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['U1_sigma'][0]**2
        self.wU2 = np.ones(self.nsize)/self.hdu[self.stats_hdu].data['U2_sigma'][0]**2

    def flag_data(self):
        """ Flag the data based on the dayflag and sundist """
        self.wI1[self.flag != 0] = 0
        self.wI2[self.flag != 0] = 0
        self.wQ1[self.flag != 0] = 0
        self.wQ2[self.flag != 0] = 0
        self.wU1[self.flag != 0] = 0
        self.wU2[self.flag != 0] = 0

    def calc_parallactic_angle(self):
        self.pa = pa(np.mod(self.ra[::100]*180./np.pi,360),self.dec[::100]*180./np.pi,self.mjd[::100],CBASS_LON,CBASS_LAT)
        #self.pa[self.pa < 0] += 360
        self.pa = np.interp(self.mjd,self.mjd[::100],self.pa, period=180)*np.pi/180.
        #self.pa[self.pa > np.pi] -= 2*np.pi

    @staticmethod
    def calc_empty_offsets(flag, offset_length=100):
        """ Find where offsets are completely masked and remove them """
        flag_steps = flag.reshape(-1,offset_length)
        flag_steps = np.repeat(np.sum(flag_steps,axis=1), offset_length) 

        good_offsets = (flag_steps < offset_length*0.1)
        return good_offsets

    def save_to_hdf5(self, filename):  
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if os.path.exists(filename):
            os.remove(filename)

        h = h5py.File(filename,'w')
        # h.create_dataset('I1',data=self.I1)
        # h.create_dataset('I2',data=self.I2)
        # h.create_dataset('Q1',data=self.Q1)
        # h.create_dataset('Q2',data=self.Q2)
        # h.create_dataset('U1',data=self.U1)
        # h.create_dataset('U2',data=self.U2)
        # h.create_dataset('wI1',data=self.wI1)
        # h.create_dataset('wI2',data=self.wI2)
        # h.create_dataset('wQ1',data=self.wQ1)
        # h.create_dataset('wQ2',data=self.wQ2)
        # h.create_dataset('wU1',data=self.wU1)
        # h.create_dataset('wU2',data=self.wU2)
        # h.create_dataset('flag',data=self.flag)
        # h.create_dataset('dayflag',data=self.dayflag)
        # h.create_dataset('sundist',data=self.sundist)
        h.create_dataset('mjd',data=self.mjd)
        h.create_dataset('az',data=self.az)
        h.create_dataset('el',data=self.el)
        h.create_dataset('ra',data=self.ra)
        h.create_dataset('dec',data=self.dec)

        for k,v in self.extra_data.items():
            h.create_dataset(k,data=v)

        h.close()

    @property
    def tod(self):
        tod = np.zeros(self.nsize) 
        mask = self.flag == 0
        tod[mask] = (self.I1[mask]*self.wI1[mask] + self.I2[mask]*self.wI2[mask])/(self.wI1[mask] + self.wI2[mask])
        tod[mask] -= np.nanmedian(tod[mask]) 
        return tod
    @property
    def tod_iqu(self):
        tod = np.zeros((3,self.nsize))
        mask = (self.flag == 0)
        tod[0,mask] = (self.I1[mask]*self.wI1[mask] + self.I2[mask]*self.wI2[mask])/(self.wI1[mask] + self.wI2[mask])
        tod[1,mask] =  self.Q1[mask]#*self.wQ1[mask] + self.Q2[mask]*self.wQ2[mask])/(self.wQ1[mask] + self.wQ2[mask])
        tod[2,mask] =  self.U1[mask]#*self.wU1[mask] + self.U2[mask]*self.wU2[mask])/(self.wU1[mask] + self.wU2[mask])
        #tod -= np.nanmedian(tod,axis=1)[:,None]
        tod_offsets = np.reshape(tod,(3,-1,self.offset_length)) 
        tod_offsets -= np.nanmedian(tod_offsets,axis=2)[:,:,None]
        del tod
        tod = tod_offsets.flatten()
        del tod_offsets 
        return tod

    @property
    def weights(self):
        weights = self.wI1[:self.nsize] + self.wI2[:self.nsize]
        return weights
    
    @property
    def weights_iqu(self):
        weights = np.zeros((3,self.nsize))
        weights[0,:] = self.wI1[:self.nsize] + self.wI2[:self.nsize]
        weights[1,:] = self.wQ1[:self.nsize] #+ self.wQ2[:self.nsize]
        weights[2,:] = self.wU1[:self.nsize] #+ self.wU2[:self.nsize]

        weights = weights.flatten()

        return weights

    @property
    def pointing(self):
        pointing = hp.ang2pix(self.nside,np.pi/2 - self.dec[:self.nsize],self.ra[:self.nsize]).astype(int)
        return pointing
    
    @property
    def pointing_iqu(self):
        pointing = np.zeros((3,self.nsize))
        pointing[:,:] = (hp.ang2pix(self.nside,np.pi/2 - self.dec[:self.nsize],self.ra[:self.nsize]).astype(int))[None,:]
        pointing[1,:] += 12*self.nside**2
        pointing[2,:] += 2*12*self.nside**2
        return pointing.flatten()
    
    @property
    def obsid_array(self):
        obsids = np.ones(self.nsize)*self.obsid
        return obsids
    
    @property
    def obsid_array_iqu(self):
        obsids = np.ones((3,self.nsize))*self.obsid
        return obsids.flatten()
    
    @property
    def special_weights(self):
        """ This is where we store the parallactic angle weights for destriping"""
        special_weights = np.ones(self.nsize)
        return special_weights

    @property
    def special_weights_iqu(self):
        """ This is where we store the parallactic angle weights for destriping"""
        pa = self.pa
        special_weights = np.ones((3,self.nsize))
        special_weights[1,:] =  np.cos(2*pa)
        special_weights[2,:] = -np.sin(2*pa)
        special_weights_rot = np.ones((3,self.nsize))
        special_weights_rot[1,:] = np.sin(2*pa)
        special_weights_rot[2,:] = np.cos(2*pa)
        special_weights_pa = np.ones((3,self.nsize)) 
        special_weights_pa[1,:] = 2
        special_weights_pa[2,:] = 3
        return special_weights.flatten(),special_weights_rot.flatten(),special_weights_pa.flatten() 
    

def read_comap_data(_filelist, offset_length=100, ifile_start=0, ifile_end=None): 
    if isinstance(ifile_end, type(None)):
        ifile_end = len(_filelist)

    if rank == 0:
        filelist = tqdm(_filelist[ifile_start:ifile_end])
    else:
        filelist = _filelist[ifile_start:ifile_end]

    ncount = 0 
    for filename in filelist:
        ncount += CBASSData.get_size(filename, offset_length=offset_length)

    # Create the arrays
    tod = np.zeros(ncount,dtype=np.float64)
    weights = np.zeros(ncount,dtype=np.float64)
    pointing = np.zeros(ncount,dtype=np.int64)
    obsid = np.zeros(ncount,dtype=np.int64)
    flags = np.zeros(ncount,dtype=np.int64)
    special_weights = np.zeros(ncount,dtype=np.float64)

    # Loop over the files
    nstart = 0

    all_cbass_data = [] 
    for i,filename in enumerate(filelist):
        cbass_data = CBASSData(filename,obsid=i+ifile_start,offset_length=offset_length)
        nend = nstart + cbass_data.nsize

        tod[nstart:nend] = cbass_data.tod
        weights[nstart:nend] = cbass_data.weights 
        pointing[nstart:nend] = cbass_data.pointing
        obsid[nstart:nend] = cbass_data.obsid_array
        flags[nstart:nend] = cbass_data.flag[:cbass_data.nsize]
        special_weights[nstart:nend] = cbass_data.special_weights

        all_cbass_data += [cbass_data]
        nstart = nend

    good_offsets = CBASSData.calc_empty_offsets(flags, offset_length=offset_length)
    return tod[good_offsets], weights[good_offsets], pointing[good_offsets], obsid[good_offsets], special_weights[good_offsets], all_cbass_data



# def index_replace(array1, array2):
#     # Create a dictionary with values and corresponding indices
#     value_to_index = {value: index for index, value in enumerate(array1)}
#     # Replace values in array2 with corresponding indices from array1
#     array3 = np.vectorize(value_to_index.get)(array2)
    
#     return array3

def index_replace(array1, array2):
    # Argsort array1, get sorted indices
    sort_indices = np.argsort(array1)

    # Create an inverse sort index array
    inv_sort_indices = np.empty_like(sort_indices)
    inv_sort_indices[sort_indices] = np.arange(sort_indices.size)

    # Sort array1 and array2 along the indices
    sorted_array1 = array1[sort_indices]
    sorted_array2 = np.searchsorted(sorted_array1, array2)

    # Create array3 by selecting elements from inverse sort index array
    array3 = inv_sort_indices[sorted_array2]

    return array3

def find_unique_values(data):
    # Perform the gather operation
    all_data = comm.allgather(data)

    # Flatten the list of lists to a single list
    all_data = [item for sublist in all_data for item in sublist]

    # Find the unique values
    unique_values = np.unique(all_data)

    return unique_values

def median_filter(tod,medfilt_stepsize):
    """
    """
    if tod.size > 2*medfilt_stepsize:
        z = np.concatenate((tod[::-1],tod,tod[::-1]))
        filter_tod = np.array(medfilt.medfilt(z.astype(np.float64),np.int32(medfilt_stepsize)))[tod.size:2*tod.size]
    else:
        filter_tod = np.ones(tod.size)*np.nanmedian(tod)

    return filter_tod[:tod.size]
def getFlatPixels(x, y,wcs,nxpix,nypix, return_xy=False):
    """
    Convert sky angles to pixel space
    """ 
    if isinstance(wcs, type(None)):
        raise TypeError( 'No WCS object declared')
        return
    else:
        pixels = wcs.wcs_world2pix(x+wcs.wcs.cdelt[0]/2.,
                                   y+wcs.wcs.cdelt[1]/2.,0)
        pflat = (pixels[0].astype(int) + nxpix*pixels[1].astype(int)).astype(int)
            

        # Catch any wrap around pixels
        pflat[(pixels[0] < 0) | (pixels[0] > nxpix)] = -1
        pflat[(pixels[1] < 0) | (pixels[1] > nypix)] = -1
    if return_xy:
        return pflat,pixels
    else:
        return pflat
def GetFeeds(file_feeds : np.ndarray, selected_feeds : np.ndarray):
    """
    Calculate the index position of each feed in the file
    """

    # Feed indices in the file in the shape of the file
    feed_indices = np.array([np.argmin(np.abs(f-file_feeds)) for i,f in enumerate(selected_feeds)])
    distances = np.array([np.abs(f-file_feeds[feed_indices[i]]) > 0 for i, f in enumerate(selected_feeds)])
    gd = (distances == 0)
    feed_indices = feed_indices[gd]
    
    # Feed indices for the output tod shape, but the same length as the file indices 
    output_indices = np.array([np.argmin(np.abs(f-selected_feeds)) for i,f in enumerate(file_feeds)])
    distances = np.array([np.abs(f-selected_feeds[output_indices[i]]) > 0 for i, f in enumerate(file_feeds)])
    gd = (distances == 0)
    output_indices = output_indices[gd]
    return feed_indices, output_indices 

def countDataSize(filename, Nfeeds, offset_length,level3='.'):
    """
    Opens each datafile and determines the number of samples
    Uses the features to select the correct chunk of data
    """
    info = {}
    try:
        d = h5py.File(filename,'r')
    except:
        print(filename)
        return info

    N = 0
    scan_edges = d['averaged_tod/scan_edges'][:]
    for (start,end) in scan_edges:
        N += int((end-start)//offset_length * offset_length)

    d.close()

    info['datasize'] = N*1.
    N = N*Nfeeds
    info['N']=int(N)

    return info


def get_tod(filename,datasize,offset_length=50,selected_feeds=[1],feed_weights=1,iband=0,level3='.',calibration=False, calibrator='TauA'):
    """
    Want to select each feed and average the data over some frequency range
    """

    d = h5py.File(filename,'r')
    dset     = d['averaged_tod/tod']
    #dset  = d['averaged_tod/tod_original']
    az_dset  = d['spectrometer/pixel_pointing/pixel_az']
    el_dset  = d['spectrometer/pixel_pointing/pixel_el']
    wei_dset = d['averaged_tod/weights']
    try:
        spike_dset = d['spikes/spike_mask'][...]
    except KeyError:
        print('LOOK HERE FOR BAD FILE!!!!!', filename)
    file_feeds = d['spectrometer/feeds'][...]
    scan_edges = d['averaged_tod/scan_edges'][...]*1
    if calibration:
        try:
            cal_factors = d[f'astro_calibration/{calibrator}_cal_factors'][...]
        except KeyError:
            print('LOOK HERE FOR BAD FILE!!!!!', filename)

    else:
        cal_factors = np.ones((dset.shape[0],dset.shape[1])) #
    
    file_feed_index, output_feed_index = GetFeeds(file_feeds, selected_feeds) # Length of nfeeds in file 

    tod     = np.zeros((len(selected_feeds), datasize))
    weights = np.zeros((len(selected_feeds), datasize))
    az      = np.zeros((len(selected_feeds), datasize))
    el      = np.zeros((len(selected_feeds), datasize))
    feedid  = np.zeros((len(selected_feeds), datasize))
    obsid   = os.path.basename(filename).split('-')[1]

    # Read in the stats too: Feed, Band, Time, Stat[White, Red, Alpha] 
    if len(d['fnoise_fits/fnoise_fit_parameters'].shape) == 3:
        print(filename)
        return tod.flatten(), weights.flatten(),az.flatten(), el.flatten(), feedid.flatten().astype(int)
    fnoise = d['fnoise_fits/fnoise_fit_parameters'][...]

    # Read in data from each feed
    for file_feed, output_feed in zip(file_feed_index, output_feed_index):
        #print(f'Calibration factors {file_feeds[file_feed]} {cal_factors[file_feed,iband]}')
        tod_file = dset[file_feed,iband,:]/cal_factors[file_feed,iband]
        weights_file = wei_dset[file_feed,iband,:]*cal_factors[file_feed,iband]**2

        spikes_file  = spike_dset[file_feed,iband,:]
        az_file      = az_dset[file_feed,:]
        el_file      = el_dset[file_feed,:]
        feedid[output_feed] = file_feeds[file_feed]
        weights_file[spikes_file] = 0


        # then the data for each scan
        last = 0
        for iscan,(start,end) in enumerate(scan_edges):

            # Check if the noise estimate is bad
            #print(file_feed, iscan, np.max(fnoise[file_feed,iband,:,1]), fnoise[file_feed,iband,:,1])
            if any([~np.isfinite(fnoise_stat) for fnoise_stat in fnoise[file_feed,iband,iscan]]) | \
                all([fnoise_stat == 0 for fnoise_stat in fnoise[file_feed,iband,iscan]]) | \
               (np.min(fnoise[file_feed,iband,:,2]) < -5.25) | (np.max(fnoise[file_feed,iband,:,1]) > 600e-3):
                print('Skipping scan %d for feed %d in %s because of bad noise estimate'%(iscan,file_feeds[file_feed],obsid),all([fnoise_stat == 0 for fnoise_stat in fnoise[file_feed,iband,iscan]]), any([~np.isfinite(fnoise_stat) for fnoise_stat in fnoise[file_feed,iband,iscan]]))
                continue # i.e., set the weight to zero
                        
            N = int((end-start)//offset_length * offset_length)
            #tod_file[start:start+N] -= median_filter(tod_file[start:start+N],int(30./0.02))

            # Set first and last 10% of data to 0 weight to remove dithering effects
            Nten = int(N*0.1) 
            weights_file[start:start+Nten] = 0
            weights_file[start+N-Nten:start+N] = 0
            
            tod[output_feed,last:last+N]     = tod_file[start:start+N]
            weights[output_feed,last:last+N] = weights_file[start:start+N] 
            az[output_feed,last:last+N]      = az_file[start:start+N]
            el[output_feed,last:last+N]      = el_file[start:start+N]
            last += N
                
    d.close()

    return tod.flatten(), weights.flatten(),az.flatten(), el.flatten(), feedid.flatten().astype(int)


def read_pixels(filename,datasize,offset_length,selected_feeds,map_info,level3='.'):
    """
    Reads data
    """


    d = h5py.File(filename,'r')

    # --- Feed position indices can change
    file_feed_index, output_feed_index = GetFeeds(d['spectrometer/feeds'][...], selected_feeds)
    
    # We store all the pointing information
    x  = d['spectrometer/pixel_pointing/pixel_ra'][file_feed_index,:]
    y  = d['spectrometer/pixel_pointing/pixel_dec'][file_feed_index,:]
    
    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']
    scan_edges = d['averaged_tod/scan_edges'][...]
    pixels = np.zeros((x.shape[0], datasize))
    last = 0
    for iscan, (start,end) in enumerate(scan_edges):
        N = int((end-start)//offset_length * offset_length)
        N = int((end-start)//offset_length * offset_length)
        end = start+N
        xc = x[:,start:end]
        yc = y[:,start:end]
        ycshape = yc.shape
        # convert to Galactic
        if 'GLON' in wcs.wcs.ctype[0]:
            rot    = hp.rotator.Rotator(coord=['C','G'])
            
            gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
            xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi
        pixels[:,last:last+N] = np.reshape(getFlatPixels(xc.flatten(),
                                                         yc.flatten(),
                                                         wcs,
                                                         nxpix,
                                                         nypix),ycshape)
        last += N
    d.close()
    return pixels 

def read_pixels_healpix(filename,datasize,offset_length,selected_feeds,map_info,level3='.',nside=4096):
    """
    Generate healpix pixels 
    """


    d = h5py.File(filename,'r')

    # --- Feed position indices can change
    file_feed_index, output_feed_index = GetFeeds(d['spectrometer/feeds'][...], selected_feeds)
    
    # We store all the pointing information
    x  = d['spectrometer/pixel_pointing/pixel_ra'][file_feed_index,:]
    y  = d['spectrometer/pixel_pointing/pixel_dec'][file_feed_index,:]
    
    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']
    scan_edges = d['averaged_tod/scan_edges'][...]
    pixels = np.zeros((x.shape[0], datasize))
    last = 0
    for iscan, (start,end) in enumerate(scan_edges):
        N = int((end-start)//offset_length * offset_length)
        N = int((end-start)//offset_length * offset_length)
        end = start+N
        xc = x[:,start:end]
        yc = y[:,start:end]
        ycshape = yc.shape
        # convert to Galactic
        if 'GLON' in wcs.wcs.ctype[0]:
            rot    = hp.rotator.Rotator(coord=['C','G'])
            gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
            xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi
        #print(xc,yc)
        try:
            pixels[:,last:last+N] = hp.ang2pix(nside, (90-yc.flatten())*np.pi/180.,xc.flatten()*np.pi/180).reshape(ycshape)
        except AttributeError:
            print(filename, N, xc, yc)
        last += N
    d.close()
    return pixels 

def read_comap_data(filelist,map_info,feed_weights=None,iband=0,offset_length=50,feeds=[i+1 for i in range(19)], calibration=False, calibrator='TauA', healpix=False):
    """
    """
    Nfeeds = len(feeds)
    if isinstance(feed_weights,type(None)):
        feed_weights = np.ones(Nfeeds)
    if rank == 0:
        _filelist = tqdm(filelist)
    else:
        _filelist = filelist
    all_info = {'N':0,'datasize':[]}
    for filename in _filelist:
        info = countDataSize(filename, Nfeeds, offset_length)
        all_info['N'] += info['N']
        all_info['datasize'] += [int(info['datasize'])]

    # create data containers
    tod      = np.zeros(all_info['N'])
    weights  = np.zeros(all_info['N'])
    pointing = np.zeros(all_info['N'],dtype=int)
    az       = np.zeros(all_info['N'])
    el       = np.zeros(all_info['N'])
    feedid   = np.zeros(all_info['N'],dtype=int)
    obsids   = np.zeros(all_info['N'],dtype=int)
    last = 0

    if rank == 0:
        _filelist = tqdm(filelist)
    else:
        _filelist = filelist

    for ifile,filename in enumerate(_filelist):
        obsid = int(os.path.basename(filename).split('-')[1])
        _tod, _weights,_az,_el, _feedid = get_tod(filename,
                                             all_info['datasize'][ifile],
                                             offset_length=offset_length,
                                             selected_feeds=feeds,
                                                 feed_weights=feed_weights,
                                             iband=iband, calibration=calibration,
                                             calibrator=calibrator)
        if healpix:
            _pointing = read_pixels_healpix(filename,
                                 all_info['datasize'][ifile],
                                 offset_length,
                                 feeds,
                                 map_info)
        else:
            _pointing = read_pixels(filename,
                                all_info['datasize'][ifile],
                                offset_length,
                                feeds,
                                map_info)

        N = _tod.size
        tod[last:last+N] = _tod
        weights[last:last+N] = _weights
        az[last:last+N] = _az
        el[last:last+N] = _el
        feedid[last:last+N] = _feedid
        pointing[last:last+N] = _pointing.flatten()
        obsids[last:last+N]  = obsid
        last += N


    mask = ~np.isfinite(tod)
    tod[mask] = 0
    weights[mask] = 0
    
    zero_mask = (weights != 0).astype(float)
    cut_empty_offsets = np.sum(zero_mask.reshape((zero_mask.size//offset_length, offset_length)),axis=1) 
    cut_empty_offsets = np.repeat(cut_empty_offsets, offset_length) 
    cut_empty_offsets = (cut_empty_offsets != 0) 
    
    tod = tod[cut_empty_offsets]
    weights=weights[cut_empty_offsets]
    pointing = pointing[cut_empty_offsets]
    az = az[cut_empty_offsets]
    el = el[cut_empty_offsets]
    feedid = feedid[cut_empty_offsets] 
    obsids = obsids[cut_empty_offsets] 
    weights[~np.isfinite(weights)] = 0

    if True:
        from matplotlib import pyplot
        from matplotlib.lines import Line2D
        import sys 
        samples = np.arange(tod.size)
        handles = [] 
        for obsid in np.unique(obsids):
            select= obsids == obsid
            plt = pyplot.plot(samples[select],tod[select],',',label=f'{obsid}')
            handles += [Line2D([0], [0], color=plt[0].get_color(), linestyle='None', marker='o', label=f'{obsid}')]
            legend = pyplot.legend(handles=handles, prop={'size': 6})
            pyplot.savefig(f'tod_figures/rank{rank:03d}_test.png')
            pyplot.close() 
    #samples = np.arange(tod.size)
    #for obsid in np.unique(obsids):
    #   select= obsids == obsid
    #   pyplot.plot(samples[select],az[select],',',label=f'{obsid}')
    #pyplot.legend(prop={'size': 6})
    #pyplot.savefig(f'tod_figures/az_rank{rank:03d}_test.png')
    #pyplot.close() 

    #comm.barrier()
    #sys.exit()
    for obsid in np.unique(obsids):
        for feed in np.unique(feedid):
            select= (feedid == feed )& (obsids == obsid)
            if len(tod[select]) < 100:
                continue
            #pmdl = np.poly1d(np.polyfit(az[select],tod[select],1))
            #tod[select] -= pmdl(az[select])
            continue
    #         pyplot.plot(samples[select],tod[select],',')
    # pyplot.xlim(0,50000)
    # #pyplot.legend()
    # pyplot.savefig('test.png')
    # stop

    remapping_array = np.unique(pointing)
    remapping_array = find_unique_values(remapping_array)
    pointing = index_replace(remapping_array, pointing)
    # share the remapping_array of each node to to all other nodes 


    return tod, weights, pointing, remapping_array.astype(int), az, el, feedid, obsids
