from comancpipeline.Tools import pysla
import numpy as np
import healpy as hp

def RotatePhi(skyVec, objRa):
    outVec = skyVec*0.
    # Rotate first RA
    outVec[:,0] =  skyVec[:,0]*np.cos(objRa*np.pi/180.) + skyVec[:,1]*np.sin(objRa*np.pi/180.) 
    outVec[:,1] = -skyVec[:,0]*np.sin(objRa*np.pi/180.) + skyVec[:,1]*np.cos(objRa*np.pi/180.) 
    outVec[:,2] =  skyVec[:,2]
    return outVec

def RotateTheta(skyVec, objDec):
    outVec = skyVec*0.
    # Rotate first Dec
    outVec[:,0] =  skyVec[:,0]*np.cos(objDec*np.pi/180.) + skyVec[:,2]*np.sin(objDec*np.pi/180.) 
    outVec[:,1] =  skyVec[:,1]
    outVec[:,2] = -skyVec[:,0]*np.sin(objDec*np.pi/180.) + skyVec[:,2]*np.cos(objDec*np.pi/180.) 
    return outVec


def RotateR(skyVec, objPang):
    outVec = skyVec*0.
    # Rotate first pang
    outVec[:,0] =  skyVec[:,0]
    outVec[:,1] =  skyVec[:,1]*np.cos(objPang*np.pi/180.) + skyVec[:,2]*np.sin(objPang*np.pi/180.) 
    outVec[:,2] = -skyVec[:,1]*np.sin(objPang*np.pi/180.) + skyVec[:,2]*np.cos(objPang*np.pi/180.) 
    return outVec

def Rotate(ra, dec, r0, d0, p0):
    """
    Rotate coordinates to be relative to some ra/dec and sky rotation pang
    
    All inputs in degrees

    """
    skyVec = hp.ang2vec((90.-dec)*np.pi/180., ra*np.pi/180.)

    outVec = RotatePhi(skyVec, r0)
    outVec = RotateTheta(outVec, d0)
    outVec = RotateR(outVec, p0)

    _dec, _ra = hp.vec2ang(outVec)
    _dec = (np.pi/2. - _dec)*180./np.pi
    _ra = _ra * 180./np.pi

    _ra[_ra > 180] -= 360.
    return _ra, _dec


def rdplan(mjd, planet, lon, lat, degrees=True):
    """
    Approximate topocentric apparent (ra,dec) and angular size of planet.

    args:
    mjd (array-like, double)
    planet (int)
    lon (array-like, double)
    lat (array-like, double)
    
    kwargs:
    degrees (bool, default=True)

    Notes: 
    planet = 1 Mercury
           = 2 Venus
           = 3 Moon
           = 4 Mars
           = 5 Jupiter
           = 6 Saturn 
           = 7 Uranus
           = 8 Neptune
           = 9 Pluto
    else   =   Sun
    """

    if not degrees:
        c = np.pi/180.
    else:
        c = 1.

    assert isinstance(planet, int), 'Error: Planet is not type int'

    return pysla.rdplan(mjd, planet, lon*c, lat*c)
    
def planet(mjd, planet):
    """
    Approximate heliocentric position and velocity of a planet.

    args:
    mjd (array-like, double)
    planet (int)

    Notes: 
    planet = 1 Mercury
           = 2 Venus
           = 3 Moon
           = 4 Mars
           = 5 Jupiter
           = 6 Saturn 
           = 7 Uranus
           = 8 Neptune
           = 9 Pluto

    """
    assert isinstance(planet, int), 'Error: Planet is not type int'

    return pysla.planet(mjd, planet)

def getPlanetPosition(source, lon, lat, mjdtod):
    """
    Get ra, dec and earth-source distance

    args:
    source name (e.g. JUPITER)
    lon : longitude of telescope (degrees)
    lat : latitude of telescope (degrees)
    mjdtod : Modified Julian date (arraylike)
    """

    if 'JUPITER' in source.upper():
        r0, d0, jdia = rdplan(mjdtod, 5, lon, lat)
        jdist = planet(mjdtod, 5)
        edist = planet(mjdtod, 3)
        rdist = np.sqrt(np.sum((jdist[:3,:] - edist[:3,:])**2,axis=0))

        r0 = np.mean(r0)*180./np.pi
        d0 = np.mean(d0)*180./np.pi
        dist = np.mean(rdist)
    else:
        r0, d0, dist = 0, 0, 0

    return r0, d0, dist


def h2e(az, el, mjd, lon, lat, degrees=True):
    """
    Horizon to equatorial coordinates

    args:
    az - arraylike, azimuth
    el - arraylike, elevation
    mjd- arraylike, modified julian date
    lon- double, longitude
    lat- double, latitude
    """

    if not degrees:
        c = np.pi/180.
    else:
        c = 1.

    ra, dec = pysla.h2e(az*c, el*c, mjd, lon*c, lat*c)
    return ra/c, dec/c

def precess(ra, dec, mjd, degrees=True):
    """
    Precess coodinrate system to FK5 J2000.
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    """

    if not degrees:
        c = np.pi/180.
    else:
        c = 1.

    ra, dec = pysla.precess(ra*c, dec*c, mjd)
    return ra/c, dec/c

def pa(ra, dec, mjd, lon ,lat, degrees=True):
    """
    Calculate parallactic angle
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    lon- double, longitude
    lat- double, latitude

    """
    
    if not degrees:
        c = np.pi/180.
    else:
        c = 1.

    p = pysla.pa(ra*c, dec*c, mjd, lon*c, lat*c)
    return p/c

def e2g(ra, dec, degrees=True):
    """
    Equatorial to Galactic
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    """
    if not degrees:
        c = np.pi/180.
    else:
        c = 1.

    gl, gb = pysla.e2g(ra*c, dec*c)
    return gl/c, gb/c
