import numpy as np
from matplotlib import pyplot
import glob
import h5py
from tqdm import tqdm
from astropy.wcs import WCS
import os
import sys
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs


def get_map_info():
    """	
    Setup wcs object		
    
    args:
    """
    w = WCS(naxis=2)
    w.wcs.crval = [0,0]
    w.wcs.cdelt = [-1./60.,1./60.]
    w.wcs.crpix = [60,60]
    w.wcs.ctype = ['RA---CAR','DEC--CAR']
    nxpix = 120
    nypix = 120
    
    map_info = {'wcs':w,
                'nxpix':nxpix,
                'nypix':nypix}
    return map_info

def get_flat_pixels(x, y,wcs,nxpix,nypix, return_xy=False):
    """
    Convert sky angles to pixel space
    """
    if isinstance(wcs, type(None)):
        raise TypeError( 'No WCS object declared')
        return
        
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
                                                                                                                

def create_pixel_coordinates(az, el, mjd, source, map_info):
    """
    Create 1D pixel coordinates in relative az/el frame

    args:
    ra  -- ndarray(float)
    dec -- ndarray(float)
    mjd -- ndarray(float)
    source -- str : TauA/CasA/CygA
    map_infp -- dict : contains WCS information
    """
    # Get the position of the source in Az/El frame
    az_src, el_src,_,_ = Coordinates.sourcePosition(source, mjd, Coordinates.comap_longitude, Coordinates.comap_latitude)
    
    # Rotate to relative Az/El frame
    az0,el0 = Coordinates.Rotate(az,el, az_src, el_src, 0)
    
    # 1D Pixel coordinates
    pixels = get_flat_pixels(az0, el0, map_info['wcs'],map_info['nxpix'],map_info['nypix'], return_xy=False)
    
    return pixels
    
def create_map_azel(tod, pixels, map_info):
    """
    Create a map in the Az/El frame
    
    args:
    tod -- ndarray(float)
    pixels -- ndarray(float)
    source -- str : TauA/CasA/CygA
    """

    
    # create tempory maps
    sky_map = np.zeros((map_info['nxpix']*map_info['nypix']))
    hit_map = np.zeros((map_info['nxpix']*map_info['nypix']))

    N = tod.size//2 * 2
    rms = np.nanstd((tod[1:N:2]-tod[0:N:2]))/np.sqrt(2)
    
    # bin the maps!
    binFuncs.binValues(sky_map, pixels, weights=tod/rms**2)
    binFuncs.binValues(hit_map, pixels, weights=np.ones(tod.size)/rms**2)

    gd = (hit_map != 0 )
    sky_map[gd] /= hit_map[gd]

    return np.reshape(sky_map,(map_info['nypix'],map_info['nxpix'])), np.reshape(hit_map,(map_info['nypix'],map_info['nxpix']))
    

def save_maps(maps, hits, utc, obsid, source, map_info, output_dir=''):
    """
    """
    _output_dir = '{}'.format(output_dir)
    
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)
        
    win = map_info['wcs']
    w = WCS(naxis=3)
    w.wcs.crval = [a for a in win.wcs.crval] + [25e9]
    w.wcs.crpix = [a for a in win.wcs.crpix] + [0]
    w.wcs.ctype = [a for a in win.wcs.ctype] + ['FREQ']
    w.wcs.cdelt = [a for a in win.wcs.cdelt] + [2e9]
    
    header   = w.to_header()
    header['SOURCE'] = source
    header['UTC']    = utc
    header['OBSID']  = obsid
    
    hdulist  = [fits.PrimaryHDU(maps[0], header=header)]
    hdulist += [fits.ImageHDU(maps[ifeed], header=header) for ifeed in range(1,20)]
    hdulist += [fits.ImageHDU(hits[ifeed], header=header) for ifeed in range(0,20)]

    hdu = fits.HDUList(hdulist)
        
    fname = f'{_output_dir}/{obsid}_{source}.fits'
    
    hdu.writeto(fname,overwrite=True)
    

def create_maps(filename, output_dir = './', fig_dir = './',east_only=False,west_only=False):
    """
    Create a fits map centre on source
    
    args:
    filename -- str : path to the level 3 datafile
    
    kwargs:
    output_dir -- str : path to output the fits files to
    fig_dir -- str : path to output quick-look images to
    
    """

    if east_only and west_only:
        raise ValueError("Both east_only and west_only can't be true")
    
    h = h5py.File(filename,'r')
    
    # Get the observation information
    source = h['comap'].attrs['source'].split(',')[0]
    obsid  = h['comap'].attrs['obsid']
    utc    = h['comap'].attrs['utc_start']
    
    # Create the map info
    map_info = get_map_info()
    
    # Read in TODs
    az_off = np.array([ 0.81736919 , 0.88856374 , 0.75226465 , 0.83386315 , 0.78688055 , 0.82400895,
                        1.00569064,  0.90985977,  0.67139243,  0.51411407,  0.64341966,  0.93197356,
                        0.91409291,  0.68841449,  0.63825983,  0.64568386,  0.95573469,  1.24559405,
                        1.15014756, -0.11470404])/60.
    el = h['pixel_pointing']['pixel_el'][...]
    az = h['pixel_pointing']['pixel_az'][...] - az_off[:,None]/np.cos(el*np.pi/180.)
    
    tod= h['tod'][...]
    mjd= h['MJD'][...]
    start,end=h['scan_edges'][0,...]
    
    N_feeds,N_bands,N_tod = tod.shape

    # Close the HDF5 file
    h.close()
    
    # Create output maps
    maps = np.zeros((20,N_bands,map_info['nypix'],map_info['nxpix']))
    hits = np.zeros((20,N_bands,map_info['nypix'],map_info['nxpix']))
    
    for ifeed in range(N_feeds):
        az_scan = az[ifeed,start:end]
        el_scan = el[ifeed,start:end]
        mjd_scan= mjd[start:end]
        select  = np.ones(az_scan.size,dtype=bool)
        if east_only:
            az_speed = np.gradient(az_scan)
            select = (az_speed > 0)
        if west_only:
            az_speed = np.gradient(az_scan)
            select = (az_speed < 0)
            
        if len(mjd_scan[select]) == 0:
            continue
        pixels = create_pixel_coordinates(az_scan[select], 
                                          el_scan[select],
                                          mjd_scan[select],
                                          source, map_info)
        for iband in range(N_bands):
            tod_scan = tod[ifeed,iband,start:end]
            tod_scan -= np.nanmedian(tod_scan)
            maps[ifeed,iband], hits[ifeed,iband] = create_map_azel(tod_scan[select], pixels, map_info)
    save_maps(maps, hits, utc, obsid, source, map_info, output_dir)
    #save_figures(maps,hits,map_info,fig_dir)

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    filelist = np.array(glob.glob('mock_level3_TauA_update2/level3_*'))

    N_files = len(filelist)
    idx = (np.sort(np.mod(np.arange(N_files),size)) == rank)

    if rank == 0:
        floop = tqdm(filelist[idx])
    else:
        floop = filelist[idx]

    for filename in floop:
        try:
            create_maps(filename,output_dir='../continuum_pipeline/TauA/pointing_corrected_fits_maps/')
        except (KeyError,ValueError):
            pass
                
