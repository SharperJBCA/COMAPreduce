# Some simple wrapper functions to make using astropy WCS functions a bit easier
import numpy as np
from matplotlib import pyplot

from astropy import wcs
from astropy.io import fits

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
    w.wcs.cdelt = np.array([-cdelt[0], cdelt[1]])
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
