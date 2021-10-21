# Some simple wrapper functions to make using astropy WCS functions a bit easier
import numpy as np
from matplotlib import pyplot

from astropy import wcs
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs
from astropy.coordinates import SkyCoord


def xlim_proj(w,x0,x1):
    pyplot.xlim(w.world_to_pixel(SkyCoord(x0,0,frame='galactic',unit='deg'))[0],\
                w.world_to_pixel(SkyCoord(x1,0,frame='galactic',unit='deg'))[0])

def ylim_proj(w,y0,y1):
    z =(w.world_to_pixel(SkyCoord(w.wcs.crval[0],y0,frame='galactic',unit='deg'))[1],\
        w.world_to_pixel(SkyCoord(w.wcs.crval[0],y1,frame='galactic',unit='deg'))[1])
    pyplot.ylim(*z)

def get_xlim_ylim(x0,y0,width,wcs,shape):

    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix, ypix,0)
    
    select = (xpix_world < (x0 + width/2.)) & (xpix_world > (x0 - width/2.)) &\
             (ypix_world < (y0 + width/2.)) & (ypix_world > (y0 - width/2.))

    xflat = np.where((np.sum(select,axis=0) > 0))[0]
    yflat = np.where((np.sum(select,axis=1) > 0))[0]

    return (min(xflat),max(xflat)), (min(yflat),max(yflat)), np.where(select.flatten())[0]


def query_disc(x0,y0,r, wcs, shape):
    """
    """
    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    rpix_world = np.sqrt((xpix_world-x0)**2*np.cos(ypix_world*np.pi/180.)**2 + (ypix_world-y0)**2)
    select = (rpix_world < r)

    return select,xpix_world[select],ypix_world[select]

def query_annullus(x0,y0,r0, r1, wcs, shape):
    """
    """
    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    rpix_world = np.sqrt((xpix_world-x0)**2*np.cos(ypix_world*np.pi/180.)**2 + (ypix_world-y0)**2)
    select = np.where((rpix_world >= r0) & (rpix_world < r1))[0]
    
    return select,xpix_world[select],ypix_world[select]

def query_slice(x0,y0,x1,y1,wcs,shape,width=None):

    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    if isinstance(width,type(None)):
        width = np.abs(wcs.wcs.cdelt[1])

    m = (y1-y0)/(x1-x0)
    yvec = m * (xpix_world-x0) + y0 

    xmid = (x1+x0)/2.
    if x1 != x0:
        xwidth=np.abs(x1-x0)/2.
    else:
        xwidth = width
    ymid = (y1+y0)/2.
    if y1 != y0:
        ywidth=np.abs(y1-y0)/2.
    else:
        ywidth = width

    select = (np.abs(yvec-ypix_world) < width) & (np.abs(ypix_world-ymid) < ywidth)  & (np.abs(xpix_world-xmid) < xwidth)  
    angular_dist = Coordinates.AngularSeperation(x0,y0,xpix_world[select], ypix_world[select])
    return select,xpix_world[select], ypix_world[select], angular_dist


def Info2WCS(naxis, cdelt, crval, ctype=['RA---TAN', 'DEC--TAN']):
    """
    Define a wcs object

    args:
    naxis = [npix_x, npix_y]
    cdelt = [cd_x, cd_y] (degrees)
    crval = [cv_x, cv_y] (degrees)

    kwargs:
    ctype : projection (Default: ['RA---TAN','DEC--TAN'])

    returns:
    wcs object
    """
    # Setup 2D wcs object
    w = wcs.WCS(naxis=2)


    w.wcs.crpix = [naxis[0]/2.+1, naxis[1]/2.+1]
    w.wcs.cdelt = np.array([cdelt[0], cdelt[1]])
    w.wcs.crval = [crval[0], crval[1]]
    w.wcs.ctype = ctype

    return w

def ang2pix(naxis, cdelt, crval, theta, phi, ctype=['RA---TAN', 'DEC--TAN']):
    """
    Sky coordinates to pixel coordinates

    args:
    naxis = [npix_x, npix_y]
    cdelt = [cd_x, cd_y] (degrees)
    crval = [cv_x, cv_y] (degrees)
    theta : latitude (arraylike, degrees)
    phi   : longitude (arraylike, degrees)

    kwargs:
    ctype : projection (Default: ['RA---TAN','DEC--TAN'])

    returns
    pixels (arraylike, int)
    """

    
    # Setup 2D wcs object
    w = Info2WCS(naxis, cdelt, crval, ctype=ctype)
    
    # Generate pixel coordinates
    pixcrd = np.floor(np.array(w.wcs_world2pix(phi, theta, 0))).astype('int64')

    bd = ((pixcrd[0,:] < 0) | (pixcrd[1,:] < 0)) | ((pixcrd[0,:] >= naxis[0]) | (pixcrd[1,:] >= naxis[1])) 

    pmax, pmin = (crval[0] + cdelt[0]*naxis[0]), (crval[0] - cdelt[0]*naxis[0])
    tmax, tmin = (crval[1] + cdelt[1]*naxis[1]), (crval[1] - cdelt[1]*naxis[1])
    cbd = (phi > pmax) | (phi <= pmin+1) | (theta <= tmin+1) | (theta > tmax)
    pix = pixcrd[0,:]*int(naxis[1]) + pixcrd[1,:]
    pix = pix.astype('int')

    pix[bd] = -1


    npix = int(naxis[0]*naxis[1])

    return pix

def pix2ang(naxis, cdelt, crval,  xpix, ypix, ctype=['RA---TAN', 'DEC--TAN']):
    """
    Pixel coordinates to sky coordinates

    args:
    naxis = [npix_x, npix_y]
    cdelt = [cd_x, cd_y] (degrees)
    crval = [cv_x, cv_y] (degrees)
    xpix   : longitude pixels (arraylike, int)
    ypix   : latitude pixels (arraylike, int)

    kwargs:
    ctype : projection (Default: ['RA---TAN','DEC--TAN'])

    returns
    latitude, longitude (degrees)
    """
    # Setup 2D wcs object
    w = Info2WCS(naxis, cdelt, crval, ctype=ctype)

    # Generate pixel coordinates
    pixcrd = np.array(w.wcs_pix2world(xpix, ypix, 0))


    return pixcrd[1,:], pixcrd[0,:]


def pix2ang1D(w, naxis, pix):
    """
    Pixel coordinates to sky coordinates

    args:
    naxis = [npix_x, npix_y]
    cdelt = [cd_x, cd_y] (degrees)
    crval = [cv_x, cv_y] (degrees)
    xpix   : longitude pixels (arraylike, int)
    ypix   : latitude pixels (arraylike, int)

    kwargs:
    ctype : projection (Default: ['RA---TAN','DEC--TAN'])

    returns
    latitude, longitude (degrees)
    """


    xpix, ypix = np.meshgrid(np.arange(naxis[0]), np.arange(naxis[1]),indexing='ij')

    # Generate pixel coordinates
    pixcrd = np.array(w.wcs_pix2world(xpix, ypix, 0))


    return pixcrd[1,:], pixcrd[0,:]



def DefineWCS(naxis=[100,100], cdelt=[1./60., 1./60.],
              crval=[0,0], ctype=['RA---TAN', 'DEC--TAN']):
    """
    Define a wcs object and sky coordinate centres of each pixel

    kwargs:
    naxis : [npix_x, npix_y]
    cdelt : [cd_x, cd_y] (degrees)
    crval : [cv_x, cv_y] (degrees)
    ctype : projection (Default: ['RA---TAN','DEC--TAN'])

    returns:
    wcs, longitude, latitude
    """

    wcs = Info2WCS(naxis, cdelt, crval, ctype=ctype)
 # Setup WCS
    xpix, ypix= np.meshgrid(np.arange(naxis[0]), np.arange(naxis[1]),indexing='ij')
    yr, xr = pix2ang(naxis, cdelt, crval,  ypix, xpix)

    xr[xr > 180] -= 360
    return wcs, xr, yr

def ang2pixWCS(wcs, ra, dec, ctype=['RA---TAN', 'DEC--TAN']):
    """
    Ang2Pix given a known wcs object

    args:
    wcs : wcs object
    ra : arraylike, degrees
    dec : arraylike, degrees

    returns:
    pixels : arraylike, int
    """

    naxis = [int((wcs.wcs.crpix[0]-1.)*2.), int((wcs.wcs.crpix[1]-1.)*2.)]


    pix = ang2pix(naxis, wcs.wcs.cdelt, wcs.wcs.crval, dec, ra, ctype=ctype).astype('int')

    return pix

def query_disc(x0,y0,r, wcs, shape):
    """
    """
    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    rpix_world = np.sqrt((xpix_world-x0)**2*np.cos(ypix_world*np.pi/180.)**2 + (ypix_world-y0)**2)
    select = (rpix_world < r)

    return select,xpix_world[select],ypix_world[select]




def udgrade_map_wcs(map_in, wcs_in, wcs_target, shape_in, shape_target,ordering='C',weights=None):
    """
    """

    if isinstance(weights,type(None)):
        weights = np.ones(map_in.size)

    # Get pixel coordinates of the input wcs
    nypix,nxpix = shape_in
    ypix,xpix = np.meshgrid(np.arange(nypix),np.arange(nxpix))
    if ordering=='C':
        ra, dec = wcs_in.wcs_pix2world(xpix.T.flatten(), ypix.T.flatten(),0)
    else:
        ra, dec = wcs_in.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    c0 = wcs_in.wcs.ctype[0].split('-')[0]
    c1 = wcs_target.wcs.ctype[0].split('-')[0]
    if c0 != c1:
        if c0 == 'GLON':
            ra,dec = Coordinates.g2e(ra,dec)

    # Convert to pixel coordinate of the output wcs
    #pix_target = ang2pixWCS(wcs_target, ra.flatten(), dec.flatten(), ctype=wcs_target.wcs.ctype)

    nypix,nxpix = shape_target
    xpix,ypix = np.floor(np.array(wcs_target.wcs_world2pix(ra.flatten(), dec.flatten(), 0))).astype('int64')
    pix_target = (xpix + ypix*nxpix).astype(np.int64)


    # Create empty target map
    map_out = np.zeros(shape_target).flatten().astype(np.float64)
    hit_out = np.zeros(shape_target).flatten().astype(np.float64)

    # Bin data to target map
    good = np.isfinite(map_in) & np.isfinite(weights)
    binFuncs.binValues(map_out, pix_target.astype(np.int64), 
                       weights=(map_in/weights).astype(np.float64), mask=good.astype(np.int64))
    binFuncs.binValues(hit_out, pix_target.astype(np.int64), 
                       weights=(1./weights).astype(np.float64),mask=good.astype(np.int64))

    return np.reshape(map_out/hit_out,shape_target), np.reshape(1./hit_out,shape_target)
