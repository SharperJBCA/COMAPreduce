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
        N += (end-start)//offset_length * offset_length
    d.close()

    info['datasize'] = N*1.
    N = N*Nfeeds
    info['N']=int(N)

    return info


def get_tod(filename,datasize,offset_length=50,selected_feeds=[1],feed_weights=1,iband=0,level3='.'):
    """
    Want to select each feed and average the data over some frequency range
    """

    d = h5py.File(filename,'r')
    dset     = d['averaged_tod/tod']
    #dset  = d['averaged_tod/tod_original']
    az_dset  = d['spectrometer/pixel_pointing/pixel_az']
    el_dset  = d['spectrometer/pixel_pointing/pixel_el']
    wei_dset = d['averaged_tod/weights']
    spike_dset = d['spikes/spike_mask'][...]
    file_feeds = d['spectrometer/feeds'][...]
    scan_edges = d['averaged_tod/scan_edges'][...]
    cal_factors = d['astro_calibration/cal_factors'][...] # np.ones((dset.shape[0],dset.shape[1])) #
    
    file_feed_index, output_feed_index = GetFeeds(file_feeds, selected_feeds) # Length of nfeeds in file 

    tod     = np.zeros((len(selected_feeds), datasize))
    weights = np.zeros((len(selected_feeds), datasize))
    az      = np.zeros((len(selected_feeds), datasize))
    el      = np.zeros((len(selected_feeds), datasize))
    feedid  = np.zeros((len(selected_feeds), datasize))
    obsid   = os.path.basename(filename).split('-')[1]

    # Read in the stats too
    fnoise = np.sqrt(d['noise_statistics/fnoise'][:,:,:,1])

    # Read in data from each feed
    for file_feed, output_feed in zip(file_feed_index, output_feed_index):
        print(f'Calibration factors {output_feed} {cal_factors[file_feed,iband]}')
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

            #if ~np.isfinite(fnoise[file_feed,iband,iscan]) | \
            #   (fnoise[file_feed,iband,iscan] > 10):
            #    print('Skipping scan %d for feed %d because of bad noise estimate'%(iscan,file_feeds[file_feed]))
            #    continue # i.e., set the weight to zero
                        
            N = int((end-start)//offset_length * offset_length)

            #tod_file[start:start+N] -= median_filter(tod_file[start:start+N],int(60./0.02))
            
            tod[output_feed,last:last+N]     = tod_file[start:start+N]
            weights[output_feed,last:last+N] = weights_file[start:start+N] # w#/feed_weights[index]
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

def read_comap_data(filelist,map_info,feed_weights=None,iband=0,offset_length=50,feeds=[i+1 for i in range(19)]):
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
                                             iband=iband)
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


    return tod, weights, pointing, az, el, feedid, obsids
