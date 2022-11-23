# Some simple wrapper functions to make using astropy WCS functions a bit easier
import numpy as np
from matplotlib import pyplot

from astropy import wcs
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs
from astropy.coordinates import SkyCoord
print('WCS CODE IS IMPORTED')

def xlim_proj(w,x0,x1):
    pyplot.xlim(w.world_to_pixel(SkyCoord(x0,0,frame='galactic',unit='deg'))[0],\
                w.world_to_pixel(SkyCoord(x1,0,frame='galactic',unit='deg'))[0])

def ylim_proj(w,y0,y1):
    z =(w.world_to_pixel(SkyCoord(w.wcs.crval[0],y0,frame='galactic',unit='deg'))[1],\
        w.world_to_pixel(SkyCoord(w.wcs.crval[0],y1,frame='galactic',unit='deg'))[1])
    pyplot.ylim(*z)


def get_flat_pixels(x, y,wcs,nxpix,nypix, return_xy=False):
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
        
        #
        pixels = wcs.wcs_pix2world(*np.unravel_index(pflat,(nypix,nxpix),order='F'),0)
        pixels[0][pixels[0] > 180] -= 360
    if return_xy:
        return pflat,pixels
    else:
        return pflat
    
def get_pixel_coordinates(wcs,shape):
    """
    Get the world coordinates of pixels
    """
    return wcs.wcs_pix2world(*np.unravel_index(np.arange(shape[0]*shape[1]),(shape),order='F'),0)


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
    x,y = Coordinates.Rotate(xpix_world,ypix_world, x0, y0, 0)

    rpix_world = np.sqrt(x**2 + y**2)
    select = (rpix_world < r)

    return select,xpix_world[select],ypix_world[select]

def query_annullus(x0,y0,r0, r1, wcs, shape):
    """
    """
    nypix,nxpix = shape
    xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
    xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

    x,y = Coordinates.Rotate(xpix_world,ypix_world, x0, y0, 0)
    rpix_world = np.sqrt(x**2+y**2)
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

def ang2pixWCS(w, phi, theta, image_shape):
    """
    Ang2Pix given a known wcs object

    args:
    wcs : wcs object
    ra : arraylike, degrees
    dec : arraylike, degrees

    returns:
    pixels : arraylike, int
    """

    # Generate pixel coordinates
    pixcrd = np.floor(np.array(w.wcs_world2pix(phi, theta, 0))).astype('int64')

    bd = ((pixcrd[0,:] < 0) | (pixcrd[1,:] < 0)) | ((pixcrd[0,:] >= image_shape[1]) | (pixcrd[1,:] >= image_shape[0])) 

    pix = pixcrd[0,:] + pixcrd[1,:]*int(image_shape[1])
    pix = pix.astype('int')
    pix[bd] = -1

    npix = int(image_shape[0]*image_shape[1])

    return pix

# def query_disc(x0,y0,r, wcs, shape):
#     """
#     """
#     nypix,nxpix = shape
#     xpix,ypix = np.meshgrid(np.arange(nxpix),np.arange(nypix))
#     xpix_world,ypix_world = wcs.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

#     rpix_world = np.sqrt((xpix_world-x0)**2*np.cos(ypix_world*np.pi/180.)**2 + (ypix_world-y0)**2)
#     select = (rpix_world < r)

#     return select,xpix_world[select],ypix_world[select]




def udgrade_map_wcs(map_in, wcs_in, wcs_target, shape_in, shape_target,ordering='C',weights=None,mask=None,mask_wcs=None):
    """
    """

    if isinstance(weights,type(None)):
        weights = np.ones(map_in.size)
    if isinstance(mask,type(None)):
        mask = np.zeros(map_in.size,dtype=bool)

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
        else:
            ra,dec = Coordinates.e2g(ra,dec)

    if not isinstance(mask_wcs, type(None)):
        #nypix,nxpix = mask.shape
        #ypix,xpix = np.meshgrid(np.arange(nypix),np.arange(nxpix))
        #ra_mask, dec_mask = wcs_in.wcs_pix2world(xpix.flatten(), ypix.flatten(),0)

        ra_mask, dec_mask = wcs_in.wcs_pix2world(xpix.T.flatten(), ypix.T.flatten(),0)
        c0 = wcs_in.wcs.ctype[0].split('-')[0]
        c1 = mask_wcs.wcs.ctype[0].split('-')[0]
        if c0 != c1:
           if c0 == 'GLON':
               ra_mask,dec_mask = Coordinates.g2e(ra_mask,dec_mask)
           else:
               ra_mask,dec_mask = Coordinates.e2g(ra_mask,dec_mask)
        #xp_mask, yp_mask = mask_wcs.wcs_world2pix(ra_mask, dec_mask, 0)
        
        pix_mask = ang2pixWCS(mask_wcs, ra_mask, dec_mask, mask.shape)
        mask_flat = mask.flatten()
        weights[~mask_flat[pix_mask]] = 0

    # Convert to pixel coordinate of the output wcs
    #pix_target = ang2pixWCS(wcs_target, ra.flatten(), dec.flatten(), ctype=wcs_target.wcs.ctype)

    nypix,nxpix = shape_target
    xpix,ypix = np.array(wcs_target.wcs_world2pix(ra.flatten(), dec.flatten(), 1))
    xpix = (xpix-0.5).astype(int)
    ypix = (ypix-0.5).astype(int)
    pix_target = (xpix + ypix*nxpix).astype(np.int64)
    
    bad = (xpix >= nxpix) | (xpix < 0) | (ypix >= nypix) | (nypix < 0)
    pix_target[bad] = -1

    # Create empty target map
    map_out = np.zeros(shape_target).flatten().astype(np.float64)
    hit_out = np.zeros(shape_target).flatten().astype(np.float64)

    # Bin data to target map
    good = np.isfinite(map_in) & np.isfinite(weights) & (weights > 0) & (pix_target != -1)
    binFuncs.binValues(map_out, pix_target[good].astype(np.int64), 
                       weights=(map_in[good]/weights[good]).astype(np.float64))#, mask=good.astype(np.int64))
    binFuncs.binValues(hit_out, pix_target[good].astype(np.int64), 
                       weights=(1./weights[good]).astype(np.float64))#,mask=good.astype(np.int64))

    # pyplot.subplot(121,projection=wcs_in)
    # pyplot.imshow(np.reshape(map_in,shape_in))
    # pyplot.contour( np.reshape(map_out/hit_out,shape_target),
    #                 colors='w',
    #                 transform=pyplot.gca().get_transform(wcs_target))
    # pyplot.subplot(122,projection=wcs_target)
    # pyplot.imshow(np.reshape(map_out/hit_out,shape_target))
    # pyplot.show()

    return np.reshape(map_out/hit_out,shape_target), np.reshape(1./hit_out,shape_target)
