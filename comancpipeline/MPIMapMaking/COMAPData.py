"""
COMAPData.py -- Read the level 3 files and return the pointing, weights and tod for destriping 
"""
import numpy as np
import h5py
from astropy import wcs
from matplotlib import pyplot
from tqdm import tqdm
import pandas as pd
from scipy import linalg as la
import healpy as hp
from comancpipeline.Tools import  binFuncs, stats,Coordinates
from comancpipeline.data import Data
from comancpipeline.Tools.median_filter import medfilt

from comancpipeline.MapMaking import MapTypes, OffsetTypes
from tqdm import tqdm 
from scipy.signal import find_peaks

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
def GetFeeds(file_feeds, selected_feeds):

    feed_indices = np.array([np.argmin(np.abs(f-file_feeds)) for i,f in enumerate(selected_feeds)])
    distances = np.array([np.abs(f-file_feeds[feed_indices[i]]) > 0 for i, f in enumerate(selected_feeds)])
    gd = (distances == 0)
    return feed_indices[gd]

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
    scan_edges = d[f'{level3}/scan_edges'][:]
    for (start,end) in scan_edges:
        N += (end-start)//offset_length * offset_length
    d.close()

    info['datasize'] = N*1.
    N = N*Nfeeds
    info['N']=N

    return info


def getTOD(filename,datasize,offset_length,Feeds,iband,level3='.'):
    """
    Want to select each feed and average the data over some frequency range
    """

    d = h5py.File(filename,'r')
    dset     = d[f'{level3}/tod']
    wei_dset = d[f'{level3}/weights']
    FeedIndex = GetFeeds(d[f'{level3}/feeds'][...], Feeds)

    az_dset = d[f'{level3}/pixel_pointing/pixel_az']

    scan_edges = d[f'{level3}/scan_edges'][...]
    tod     = np.zeros((len(FeedIndex), datasize))
    weights = np.zeros((len(FeedIndex), datasize))
    az      = np.zeros((len(FeedIndex), datasize))
    # Read in data from each feed
    for index, ifeed in enumerate(FeedIndex[:]):

        tod_file = dset[ifeed,iband,:]
        weights_file = wei_dset[ifeed,iband,:]
        az_file = az_dset[ifeed,:]

        # then the data for each scan
        last = 0
        for iscan,(start,end) in enumerate(scan_edges):
            N = int((end-start)//offset_length * offset_length)
            end = start+N
            
            try:
                #min_az = np.min(az_file[start:end])
                #max_az = np.max(az_file[start:end])
                #az_bin = 0.25
                #nbins  = int((max_az-min_az)/az_bin)
                #azedges= np.linspace(min_az,max_az,nbins+1)
                #azmids = (azedges[1:] + azedges[:-1])/2.
                #azbinned = np.histogram(az_file[start:end], azedges, weights=tod_file[start:end])[0]/\
                #           np.histogram(az_file[start:end], azedges)[0]
                #pyplot.plot(tod_file[start:end])
                #pyplot.plot(np.interp(az_file[start:end],azmids,azbinned))
                #pyplot.show()

                tod[index,last:last+N]     = tod_file[start:end] - median_filter(tod_file[start:end],20)
            except (ValueError, IndexError):
                tod[index,last:last+N]     = tod_file[start:end]
            weights[index,last:last+N] = weights_file[start:end]
            az[index,last:last+N] = az_file[start:end]
            last += N
                
    d.close()
    return tod.flatten(), weights.flatten(), az.flatten()


def readPixels(filename,datasize,offset_length,Feeds,map_info,level3='.'):
    """
    Reads data
    """


    d = h5py.File(filename,'r')

    # --- Feed position indices can change
    FeedIndex = GetFeeds(d[f'{level3}/feeds'][...], Feeds)
    
    # We store all the pointing information
    x  = d[f'{level3}/pixel_pointing/pixel_ra'][FeedIndex,:]
    y  = d[f'{level3}/pixel_pointing/pixel_dec'][FeedIndex,:]
    
    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']
    scan_edges = d[f'{level3}/scan_edges'][...]
    pixels = np.zeros((x.shape[0], datasize))
    last = 0
    for iscan, (start,end) in enumerate(scan_edges):
        N = int((end-start)//offset_length * offset_length)
        end = start+N
        xc = x[:,start:end]
        yc = y[:,start:end]
        # convert to Galactic
        if 'GLON' in wcs.wcs.ctype[0]:
            rot    = hp.rotator.Rotator(coord=['C','G'])
            gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
            xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi

        pixels[:,last:last+N] = np.reshape(getFlatPixels(xc.flatten(),
                                                         yc.flatten(),
                                                         wcs,
                                                         nxpix,
                                                         nypix),yc.shape)
        last += N
    d.close()
    return pixels 

def read_comap_data(filelist,map_info,iband=0,offset_length=50,feeds=[i+1 for i in range(19)]):
    """
    """
    Nfeeds = len(feeds)

    if rank == 0:
        filelist = tqdm(filelist)

    all_info = {'N':0,'datasize':[]}
    for filename in filelist:
        info = countDataSize(filename, Nfeeds, offset_length)
        all_info['N'] += info['N']
        all_info['datasize'] += [int(info['datasize'])]

    # create data containers
    tod      = np.zeros(all_info['N'])
    weights  = np.zeros(all_info['N'])
    pointing = np.zeros(all_info['N'],dtype=int)
    az       = np.zeros(all_info['N'])
    last = 0
    for ifile,filename in enumerate(filelist):
        _tod, _weights,_az = getTOD(filename,
                                all_info['datasize'][ifile],
                                offset_length,
                                feeds,
                                iband)
        _pointing = readPixels(filename,
                               all_info['datasize'][ifile],
                               offset_length,
                               feeds,
                               map_info)
        N = _tod.size
        tod[last:last+N] = _tod
        weights[last:last+N] = _weights
        az[last:last+N] = _az
        pointing[last:last+N] = _pointing.flatten()

        last += N

    mask = ~np.isfinite(tod)
    tod[mask] = 0
    weights[mask] = 0

    return tod, weights, pointing, az
