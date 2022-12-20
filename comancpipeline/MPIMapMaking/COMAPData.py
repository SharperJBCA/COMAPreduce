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
import os

from comancpipeline.MapMaking import MapTypes, OffsetTypes
from tqdm import tqdm 
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

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
    info = {'N':0,'datasize':0}
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
    info['N']=int(N)

    return info


def getTOD(filename,datasize,tod_name='filtered_tod',offset_length=50,Feeds=[1],feed_weights=1,iband=0,level3='.'):
    """
    Want to select each feed and average the data over some frequency range
    """

    d = h5py.File(filename,'r')
    dset     = d[f'{level3}/{tod_name}']
    dset_og  = d[f'{level3}/tod']
    az_dset  = d[f'{level3}/updated_pixel_pointing/pixel_az']
    el_dset  = d[f'{level3}/updated_pixel_pointing/pixel_el']
    ra_dset  = d[f'{level3}/updated_pixel_pointing/pixel_ra']
    dec_dset = d[f'{level3}/updated_pixel_pointing/pixel_dec']
    wei_dset = d[f'{level3}/weights']
    cal_factors=d[f'{level3}/CALFACTORS/calibration_factors'][...]
    FeedIndex = GetFeeds(np.arange(1,21), Feeds)
    pointing_feeds = d[f'{level3}/feeds'][...]
    pointFeedIndex =  GetFeeds(d[f'{level3}/feeds'][...], Feeds)
    scan_edges = d[f'{level3}/scan_edges'][...]
    tod     = np.zeros((len(FeedIndex), datasize))
    weights = np.zeros((len(FeedIndex), datasize))
    az      = np.zeros((len(FeedIndex), datasize))
    el      = np.zeros((len(FeedIndex), datasize))
    feedid  = np.zeros((len(FeedIndex), datasize))
    obsid   = os.path.basename(filename).split('-')[1]

    # Read in the stats too
    if 'fnoise_fits' in d[f'{level3}']:
        fnoise = np.sqrt(d[f'{level3}/fnoise_fits'][:,:,:,0])
        if len(d[f'{level3}/cov5s'].shape) == 3:
            cov5s  = d[f'{level3}/cov5s'][0,1,:]
        else:
            cov5s = d[f'{level3}/cov5s'][0,1]*np.ones(fnoise.shape[0])
    else:
        fnoise = None
        cov5s = None

    # Read in data from each feed
    for index, ifeed in enumerate(FeedIndex[:]):

        if not (ifeed+1) in pointing_feeds:
            continue
        pifeed = np.argmin((pointing_feeds - (ifeed+1))**2)
        tod_file = dset[ifeed,iband,:]/cal_factors[ifeed,iband]
        if len(wei_dset.shape) == 3:
            weights_file = wei_dset[ifeed,iband,:]*cal_factors[ifeed,iband]**2
        else:
            weights_file = (wei_dset[ifeed,iband])[...,None]*np.ones(tod_file.shape)
        az_file      = az_dset[ifeed,:]
        el_file      = el_dset[ifeed,:]
        ra_file      = ra_dset[ifeed,:]
        dec_file     = dec_dset[ifeed,:]
        gl_file, gb_file = Coordinates.e2g(ra_file,dec_file)
        feedid[index] = (ifeed+1)

        
        # then the data for each scan
        last = 0
        for iscan,(start,end) in enumerate(scan_edges):
            N = int((end-start)//offset_length * offset_length)
            end = start+N

            tod_temp = tod_file[start:end]
            if np.nansum(tod_temp) == 0:
                continue
            # Check some stats:
            idx_feed = np.where((pointing_feeds == (ifeed+1)))[0]
            if len(idx_feed) == 0:
                continue
            ratio = d['fnoise_fits'][idx_feed,iband,iscan,1].flatten()/d['fnoise_fits'][idx_feed,iband,iscan,0].flatten()
            alpha = d['fnoise_fits'][idx_feed,iband,iscan,2].flatten()
            t = np.arange(start,end)
            smoothed_tod = median_filter(tod_temp,250)
            pmdl = np.poly1d(np.polyfit(t, smoothed_tod, 1))
            grad = pmdl[1]*1e6
            chi2 = np.mean((smoothed_tod - pmdl(t))**2/d['fnoise_fits'][idx_feed,iband,iscan,0])

            if (ratio > 20) | (alpha < -1.5) | (np.abs(grad) > 5) | (chi2 > 0.5): # Check conditions
                continue

            if not isinstance(fnoise,type(None)):
                #if ~np.isfinite(fnoise[ifeed,iband,iscan]) | \
                #   (fnoise[ifeed,iband,iscan] > 0.4) | \
                #if (cov5s[iscan] > 0.85) | \
                #   (cov5s[iscan] == 0):
                #    continue # i.e., set the weight to zero
                pass
            
            a = az_file[start:end]
            e = el_file[start:end]
            a -= np.nanmedian(a)
            a *= np.cos(e*np.pi/180.)
            e -= np.nanmedian(e)

            minel = np.nanmin(e)
            maxel = np.nanmax(e)
            diffel = maxel-minel
            minaz = np.nanmin(a)
            maxaz = np.nanmax(a)
            diffaz= maxaz-minaz
            #d_a = gaussian_filter(np.gradient(a)/(1./50.),5)
            #d_e = gaussian_filter(np.gradient(e)/(1./50.),5)
            #maxe = np.nanmax(d_e)
            #maxa = np.nanmax(d_a)

            if diffel > 0.1:
                gd = (np.abs(a) > diffaz/2.*0.9) | (np.abs(e) > diffel/2.*0.9)
            else:
                gd = (np.abs(a) > diffaz/2.*0.9) 

            
            N2 = tod_temp.size//2 * 2
            rms = np.nanstd(tod_temp[1:N2:2]-tod_temp[0:N2:2])
            if rms == 0:
                continue
            #print(N2,rms)
            w = np.ones(tod_temp.size)*1./np.nanstd(tod_temp[1:N2:2]-tod_temp[0:N2:2])**2
            #weights_file[start:end]
            #print(np.max(np.abs(a)), diffaz/2.*0.9, np.max(np.abs(e)), diffel/2.*0.9)
            #print(minel,maxel,maxe,maxa)
            w[gd] = 0
            z = w > 0
            zsum = np.repeat(np.sum(np.reshape(z,(z.size//50,50)),axis=1),50)
            w[zsum < 45] = 0

            w[:500] = 0
            w[-100:] = 0
            #print(np.sum(w))
            if np.sum(w) == 0:
                continue

            _d = tod_temp

            _d -= median_filter(_d,1000)

            tod_mask = (w != 0) & (np.abs(gb_file[start:end]) > 1)
            if len(a[tod_mask]) > 10:
                mina,maxa = np.min(a[tod_mask]),np.max(a[tod_mask])
                mine,maxe = np.min(e[tod_mask]),np.max(e[tod_mask])
                stepsize=(4./60.)
                nbins = int((maxe-mine)/stepsize)
                aedges = np.linspace(mine,maxe,nbins+1)
                amids  = (aedges[1:]+aedges[:-1])/2.
                top = np.histogram(e[tod_mask],aedges,weights=_d[tod_mask]*w[tod_mask])[0]
                bot = np.histogram(e[tod_mask],aedges,weights=w[tod_mask])[0]
                mdl = top/bot

                if len(amids[np.isfinite(mdl)]) > 0:
                    _d -= np.poly1d(np.polyfit(amids[np.isfinite(mdl)], mdl[np.isfinite(mdl)],1))(e)

                stepsize=(4./60.)
                nbins = int((maxa-mina)/stepsize)
                aedges = np.linspace(mina,maxa,nbins+1)
                amids  = (aedges[1:]+aedges[:-1])/2.
                top = np.histogram(a[tod_mask],aedges,weights=_d[tod_mask]*w[tod_mask])[0]
                bot = np.histogram(a[tod_mask],aedges,weights=w[tod_mask])[0]
                mdl = top/bot

                if len(amids[np.isfinite(mdl)]) > 0:
                    _d -= np.poly1d(np.polyfit(amids[np.isfinite(mdl)], mdl[np.isfinite(mdl)],1))(a)
             

            tod[index,last:last+N]     = _d #tod_file[start:end]
            weights[index,last:last+N] = w#/feed_weights[index]
            az[index,last:last+N]      = az_file[start:end]
            el[index,last:last+N]      = el_file[start:end]

            last += N
                
    d.close()
    return tod.flatten(), weights.flatten(),az.flatten(), el.flatten(), feedid.flatten().astype(int)


def readPixels(filename,datasize,offset_length,Feeds,map_info,level3='.'):
    """
    Reads data
    """


    d = h5py.File(filename,'r')

    # --- Feed position indices can change
    FeedIndex = GetFeeds(np.arange(1,21), Feeds)
    
    # We store all the pointing information
    x  = d[f'{level3}/updated_pixel_pointing/pixel_ra'][FeedIndex,:]
    y  = d[f'{level3}/updated_pixel_pointing/pixel_dec'][FeedIndex,:]
    
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
        xc[~np.isfinite(xc)]=0
        yc[~np.isfinite(yc)]=0
        ycshape = yc.shape
        # convert to Galactic
        if 'GLON' in wcs.wcs.ctype[0]:
            rot    = hp.rotator.Rotator(coord=['C','G'])
            gb, gl = rot((90-yc.flatten())*np.pi/180., xc.flatten()*np.pi/180.)
            xc, yc = gl*180./np.pi, (np.pi/2-gb)*180./np.pi

        
        if isinstance(xc,float):
            pixels[:,last:last+N] = np.reshape(getFlatPixels(np.array([xc])[None,:],
                                                             np.array([yc])[None,:],
                                                             wcs,
                                                             nxpix,
                                                             nypix), ycshape)
        else:
            pixels[:,last:last+N] = np.reshape(getFlatPixels(xc.flatten(),
                                                             yc.flatten(),
                                                             wcs,
                                                             nxpix,
                                                             nypix),ycshape)
        last += N
    d.close()
    return pixels 

def read_comap_data(filelist,map_info,tod_name='filtered_tod',feed_weights=None,iband=0,offset_length=50,feeds=[i+1 for i in range(19)]):
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
        _tod, _weights,_az,_el, _feedid = getTOD(filename,
                                                 all_info['datasize'][ifile],
                                                 offset_length=offset_length,
                                                 Feeds=feeds,
                                                 tod_name=tod_name,
                                                 feed_weights=feed_weights,
                                                 iband=iband)

        _pointing = readPixels(filename,
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

    # Now just remove all the zeros
    zero_mask = (weights == 0)
    # remove just the points where a full offset is zero, not partial offsets
    zero_mask = np.repeat(np.sum(np.reshape(zero_mask,(zero_mask.shape[0]//offset_length,offset_length)),axis=-1),offset_length)== 0 
    tod = tod[zero_mask]
    weights=weights[zero_mask]
    pointing=pointing[zero_mask]
    az=az[zero_mask]
    el=el[zero_mask]
    feedid = feedid[zero_mask]
    obsids = obsids[zero_mask]

    return tod, weights, pointing, az, el, feedid, obsids
