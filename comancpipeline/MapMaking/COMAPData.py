"""
COMAPData.py -- Read the level 3 files and return the pointing, weights and tod for destriping 
"""
import numpy as np
import h5py
from tqdm import tqdm
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
import os
import healpy as hp 
from astropy.coordinates import get_sun 
from astropy.time import Time
from scipy.stats import binned_statistic

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# def index_replace(array1, array2):
#     # Create a dictionary with values and corresponding indices
#     value_to_index = {value: index for index, value in enumerate(array1)}
#     # Replace values in array2 with corresponding indices from array1
#     array3 = np.vectorize(value_to_index.get)(array2)
    
#     return array3


def parse_bit_mask(flag):
    p = np.inf 
    bit_mask_list = []
    current_flag = flag*1
    while p != 0: 
        if current_flag == 0:
            bit_mask_list.append(0)
            break
        p = int(np.floor(np.log(current_flag)/np.log(2)))
        bit_mask_list.append(p)
        current_flag -= 2**p
    return bit_mask_list


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

def transform_to_1d(x, y, wcs, nx, ny, return_xy=False):
    """
    Transforms sky coordinates to 1D image pixel coordinates.

    Args:
        x (np.array): x-coord  values.
        y (np.array): y-coord values.
        wcs (WCS): Astropy WCS object.
        nx (int): The number of pixels in the x-direction.
        ny (int): The number of pixels in the y-direction.

    Returns:
        np.array: 1D pixel coordinates.
    """
    # Transform sky coordinates to 2D pixel coordinates
    px, py = wcs.wcs_world2pix(x, y, 0)

    # Make sure pixel coordinates are integers
    px = np.floor(px+0.5).astype(float)
    py = np.floor(py+0.5).astype(float)
    
    # Some of the calculated pixel coordinates can be out of image dimensions
    # Need to make sure the pixel coordinates are within the image
    px[(px < 0) | (px > nx-1)] = np.nan
    py[(py < 0) | (py > ny-1)] = np.nan

    # Map 2D pixel coordinates to 1D array index
    index_1d = py * nx + px
    index_1d[np.isnan(index_1d)] = -1 
    index_1d = index_1d.astype(int) 
    
    if return_xy:
        return index_1d,hp.vstack([px,py])
    else:
        return index_1d

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

def get_scan_edges(d):
    try:
        scan_edges = d['averaged_tod/scan_edges'][:]
    except KeyError:
        scan_edges = [[0,0]]
    return scan_edges

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
    scan_edges = get_scan_edges(d)
    if len(scan_edges) > 0:
        for (start,end) in scan_edges:
            N += int((end-start)//offset_length * offset_length)

    d.close()

    info['datasize'] = N*1.
    N = N*Nfeeds
    info['N']=int(N)

    return info

def sun_centric_coords(ra,dec,mjd,az,el):
    """
    Convert the RA/DEC to Sun centric coordinates
    """
    # Get the sun position
    sun = get_sun(Time(mjd,format='mjd'))
    sun_ra = sun.ra.deg
    sun_dec = sun.dec.deg

    # Convert the RA/DEC to Sun centric coordinates
    rot = hp.rotator.Rotator(rot=[sun_ra,sun_dec],inv=True)
    theta,phi = (np.pi/2.-dec*np.pi/180.),ra*np.pi/180.
    theta,phi = rot(theta,phi)

    return phi, theta 

def auto_rms(tod):
    N = tod.size//2*2
    diff = tod[1:N] - tod[:-1:N]
    return np.nanstd(diff) /np.sqrt(2)

# import astropy sun position
from astropy.coordinates import get_sun

def get_sun_centric_coords(ra,dec,mjd):
    """
    Convert the RA/DEC to Sun centric coordinates
    """
    # Get the sun position
    sun = get_sun(Time(mjd[0],format='mjd'))
    sun_ra = sun.ra.deg
    sun_dec = sun.dec.deg

    # Convert the RA/DEC to Sun centric coordinates
    rot = hp.rotator.Rotator(rot=[sun_ra,sun_dec],inv=True)
    theta,phi = (np.pi/2.-dec*np.pi/180.),ra*np.pi/180.
    good_vals = np.isfinite(ra) & np.isfinite(dec)
    theta[good_vals],phi[good_vals] = rot(theta[good_vals],phi[good_vals])

    if isinstance(phi, float):
        print('WTF?!?',ra.size,dec.size,mjd.size,sun_ra,sun_dec,np.nansum(ra),np.nansum(dec))
        print(theta,phi)

    return phi, theta

# Haversine formula for a unit sphere
def haversine(theta1,phi1,theta2,phi2):
    return 2*np.arcsin(np.sqrt(np.sin((theta2-theta1)/2)**2+np.cos(theta1)*np.cos(theta2)*np.sin((phi2-phi1)/2)**2))

def read_calibration_factors(d, source):
    cal_factors = np.zeros((20,4))
    cal_factors[:,0] = d['comap'].attrs[f'{source}_calibration_factor_band0']
    cal_factors[:,1] = d['comap'].attrs[f'{source}_calibration_factor_band1']
    cal_factors[:,2] = d['comap'].attrs[f'{source}_calibration_factor_band2']
    cal_factors[:,3] = d['comap'].attrs[f'{source}_calibration_factor_band3']
    return cal_factors 


def get_tod(filename,pointing,datasize,offset_length=50,selected_feeds=[1],use_gain_filter=True, feed_weights=1,iband=0,level3='.',calibration=False, calibrator='TauA'):
    """
    Want to select each feed and average the data over some frequency range
    """

    d = h5py.File(filename,'r')
    source = d['comap'].attrs['source'].split(',')[0]

    if use_gain_filter and not source in ['TauA','CasA','CygA','jupiter']:
        dset     = d['averaged_tod/tod']
    else:
        dset  = d['averaged_tod/tod_original']
    az_dset  = d['spectrometer/pixel_pointing/pixel_az']
    el_dset  = d['spectrometer/pixel_pointing/pixel_el']
    ra_dset = d['spectrometer/pixel_pointing/pixel_ra']
    dec_dset = d['spectrometer/pixel_pointing/pixel_dec']
    wei_dset = d['averaged_tod/weights']
    mjd_dset = d['spectrometer/MJD']
    bad_feeds = d['comap'].attrs['bad_observation']
    try:
        spike_dset = d['spikes/spike_mask'][...]
        if len(spike_dset.shape) == 1:
            spike_dset = None 
    except KeyError:
        spike_dset = None
        print('LOOK HERE FOR BAD FILE!!!!!', filename)
    
    file_feeds = d['spectrometer/feeds'][...]
    if calibration:
        try:
            cal_factors = read_calibration_factors(d, calibrator)
        except KeyError:
            print('LOOK HERE FOR BAD FILE!!!!!', filename)
    else:
        cal_factors = np.ones((dset.shape[0],dset.shape[1])) #
    
    file_feed_index, output_feed_index = GetFeeds(file_feeds, selected_feeds) # Length of nfeeds in file 

    tod     = np.zeros((len(output_feed_index), datasize))
    weights = np.zeros((len(output_feed_index), datasize))
    az      = np.zeros((len(output_feed_index), datasize))
    el      = np.zeros((len(output_feed_index), datasize))
    ra     = np.zeros((len(output_feed_index), datasize))
    dec     = np.zeros((len(output_feed_index), datasize))
    feedid  = np.zeros((len(output_feed_index), datasize))
    obsid   = os.path.basename(filename).split('-')[1]

    scan_edges = get_scan_edges(d)
    if len(scan_edges) == 0:
        return tod.flatten(), weights.flatten(),az.flatten(), el.flatten(), ra.flatten(), dec.flatten(), feedid.flatten().astype(int)

    mask_map_option=False
    if mask_map_option:
        from astropy.io import fits
        mask_map = fits.open('maps/galactic/All_galactic_low_gainfilter_medianFilter400_Band00_cutoff75.fits')[0].data
        mask_map = (mask_map > 0.05) & np.isfinite(mask_map)
        mask_map = mask_map.flatten() 

    idx = np.arange(pointing.size)
    for ifeed, (file_feed, output_feed) in enumerate(zip(file_feed_index, output_feed_index)):
        feed_bit_mask = bad_feeds[file_feeds[file_feed]] 
        bad_feed = False
        for bit_value in parse_bit_mask(feed_bit_mask):
            if bit_value != 0 and bit_value != 5:#  and bit_value != 1:  # 1 is bad stats, ignore for now. 
                bad_feed = True 
                break
        #print(ifeed, file_feed, output_feed, bad_feed, parse_bit_mask(feed_bit_mask))
        if bad_feed:
            continue 

        tod_file = dset[file_feed,iband,:]/cal_factors[file_feed,iband]
        tod_file_copy = tod_file[scan_edges[0][0]:scan_edges[-1][1]]*1. 

        weights_file = np.ones(tod_file.size)/auto_rms(tod_file)**2

        az_file      = az_dset[file_feed,:]
        #az_file -= np.median(az_file)
        
        el_file      = el_dset[file_feed,:]
        ra_file, dec_file = get_sun_centric_coords(ra_dset[file_feed,:],dec_dset[file_feed,:],mjd_dset[:])
        ra_file = haversine(0,0,ra_file, dec_file) * 180.0/np.pi
        #ra_file      = ra_dset[file_feed,:]
        #dec_file     = dec_dset[file_feed,:]
        feedid[output_feed] = file_feeds[file_feed]

        if not isinstance(spike_dset,type(None)):
            spikes_file  = spike_dset[file_feed,iband,:]
            weights_file[spikes_file] = 0
        weights_file[ra_file < 10] = 0


        # mask 10% edges of az and el
        good = np.isfinite(az_file)
        edge_cut_fraction = 10
        az10 = np.percentile(az_file[good],edge_cut_fraction)
        az90 = np.percentile(az_file[good],100-edge_cut_fraction)
        el10 = np.percentile(el_file[good],edge_cut_fraction)
        el90 = np.percentile(el_file[good],100-edge_cut_fraction)
        weights_file[(az_file < az10) | (az_file > az90)] = 0
        weights_file[(el_file < el10) | (el_file > el90)] = 0

        # then the data for each scan
        last = 0
        for iscan,(start,end) in enumerate(scan_edges):                        
            N = int((end-start)//offset_length * offset_length)

            p = pointing[ifeed,last:last+N]
            #bad = (mask_map[p])
            tod_copy = tod_file[start:start+N]*1.
            #tod_copy[bad] = np.interp(p[bad],p[~bad],tod_copy[~bad])
            bad = (tod_copy == 0) 
            tod_slice = tod_file[start:start+N]
            if not source in ['TauA','CasA','CygA','jupiter']:
                tod_slice[~bad] -= median_filter(tod_copy[~bad],400)

            # Set first and last 10% of data to 0 weight to remove dithering effects
            Nten = int(N*0.1) 
            weights_file[start:start+Nten] = 0
            weights_file[start+N-Nten:start+N] = 0
                        
            tod[output_feed,last:last+N]     = tod_slice
            weights[output_feed,last:last+N] = weights_file[start:start+N] 
            az[output_feed,last:last+N]      = az_file[start:start+N]
            el[output_feed,last:last+N]      = el_file[start:start+N]
            try:
                ra[output_feed,last:last+N]      = ra_file[start:start+N]
            except IndexError:
                print(ra.shape, ra_file.shape, N, tod.shape, tod_file.shape)
            dec[output_feed,last:last+N]     = dec_file[start:start+N]
            last += N
                
    d.close()

    return tod.flatten(), weights.flatten(),az.flatten(), el.flatten(), ra.flatten(), dec.flatten(), feedid.flatten().astype(int)


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
    scan_edges = get_scan_edges(d)
    pixels = np.zeros((len(output_feed_index), datasize))
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
        if not isinstance(xc, np.ndarray):
            continue
        _pixels = np.reshape(transform_to_1d(xc.flatten(),
                                                         yc.flatten(),
                                                         wcs,
                                                         nxpix,
                                                         nypix),ycshape)
        for ifeed, (file_feed, output_feed) in enumerate(zip(file_feed_index, output_feed_index)):
            pixels[ifeed,last:last+N] = _pixels[output_feed,:]
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

def read_comap_data(filelist,map_info,feed_weights=None,iband=0,use_gain_filter=True,offset_length=50,feeds=[i+1 for i in range(19)], calibration=False, calibrator='TauA', healpix=False):
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
    ra       = np.zeros(all_info['N'])
    dec       = np.zeros(all_info['N'])
    feedid   = np.zeros(all_info['N'],dtype=int)
    obsids   = np.zeros(all_info['N'],dtype=int)
    last = 0

    if rank == 0:
        _filelist = tqdm(filelist)
    else:
        _filelist = filelist

    for ifile,filename in enumerate(_filelist):
        obsid = int(os.path.basename(filename).split('-')[1])

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

        _tod, _weights,_az,_el,_ra,_dec, _feedid = get_tod(filename,_pointing.astype(int),
                                             all_info['datasize'][ifile],
                                             offset_length=offset_length,
                                             selected_feeds=feeds,
                                             use_gain_filter=use_gain_filter,
                                                 feed_weights=feed_weights,
                                             iband=iband, calibration=calibration,
                                             calibrator=calibrator)

        try:
            pass
            #if np.min(_ra[_ra != 0]) < 79.4:
            #    print(filename, np.min(_ra[_ra != 0]), np.max(_ra[_ra != 0])   )
        except ValueError:
            print('BAD',filename)
            print('BAD',_ra.size)
            print('BAD',np.sum(_ra))
        N = _tod.size
        tod[last:last+N] = _tod
        weights[last:last+N] = _weights
        az[last:last+N] = _az
        el[last:last+N] = _el
        ra[last:last+N] = _ra
        dec[last:last+N] = _dec
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
    ra = ra[cut_empty_offsets]
    dec = dec[cut_empty_offsets]
    feedid = feedid[cut_empty_offsets] 
    obsids = obsids[cut_empty_offsets] 
    weights[~np.isfinite(weights)] = 0

    remapping_array = np.unique(pointing)
    remapping_array = find_unique_values(remapping_array)
    if healpix:
        pointing = index_replace(remapping_array, pointing)
    # share the remapping_array of each node to to all other nodes 


    return tod, weights, pointing, remapping_array.astype(int), az, el, ra, dec, feedid, obsids
