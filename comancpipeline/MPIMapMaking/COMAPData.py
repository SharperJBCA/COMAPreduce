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

from scipy.signal import find_peaks

def GetFeeds(file_feeds, selected_feeds):

    feed_indices = np.array([np.argmin(np.abs(f-file_feeds)) for i,f in enumerate(selected_feeds)])
    distances = np.array([np.abs(f-file_feeds[feed_indices[i]]) > 0 for i, f in enumerate(selected_feeds)])
    gd = (distances == 0)
    return feed_indices[gd]

def countDataSize(filename, Nfeeds, offset_length):
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
    scan_edges = d[f'{self.level3}/scan_edges'][:]
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

    scan_edges = d[f'{level3}/scan_edges'][...]
    tod     = np.zeros((len(FeedIndex), datasize))
    weights = np.zeros((len(FeedIndex), datasize))
    # Read in data from each feed
    for index, ifeed in enumerate(FeedIndex[:]):

        tod_file = dset[ifeed,iband,:]
        weights_file = wei_dset[ifeed,iband,:]
        
        # then the data for each scan
        last = 0
        for iscan,(start,end) in enumerate(scan_edges):
            N = int((end-start)//offset_length * offset_length)
            end = start+N
            tod[index,last:last+N]     = tod_file[start:end]
            weights[index,last:last+N] = weights_file[start:end]
            last += N
                
    d.close()
    return tod.flatten(), weights.flatten()


    def readPixels(filename,datasize,offset_length,Feeds,wcs,level3='.'):
        """
        Reads data
        """


        d = h5py.File(filename,'r')

        # --- Feed position indices can change
        FeedIndex = GetFeeds(d[f'{level3}/feeds'][...], Feeds)

        # We store all the pointing information
        x  = d[f'{level3}/pixel_pointing/pixel_ra'][FeedIndex,:]
        y  = d[f'{level3}/pixel_pointing/pixel_dec'][FeedIndex,:]

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

            pixels[:,last:last+N] = np.reshape(getFlatPixels(xc.flatten(),yc.flatten()),yc.shape)
            last += N
        d.close()
        return pixels 

def read_comap_data(filelist,wcs,offset_length=50,feeds=[i+1 for i in range(19)]):
    """
    """
    Nfeeds = len(feeds)

    all_info = {'N':0,'datasize':[]}
    for filename in filelist:
        info = countDataSize(filename, Nfeeds, offset_length)
        all_info['N'] += info['N']
        all_info['datasize'] += [info['datasize']]

    # create data containers
    tod      = np.zeros(all_info['N'])
    weights  = np.zeros(all_info['N'])
    pointing = np.zeros(all_info['N'],dtype=int)
    
    last = 0
    for ifile,filename in enumerate(filelist):
        _tod, _weights = getTOD(filename,
                                all_info['datasize'][ifile],
                                offset_length,
                                feeds,
                                iband)
        _pointing = readPixels(filename,
                               all_info['datasize'][ifile],
                               offset_length,
                               feeds,
                               wcs)
        N = _tod.size
        tod[last:last+N] = _tod
        weights[last:last+N] = _weights
        pointing[last:last+N] = _pointing

        last += N

    return tod, weights, pointing
