from comancpipeline.Tools import pysla
import numpy as np
import healpy as hp

# List of sources used for calibration
# Ephem objects have None, calculated on the fly
CalibratorList = {
    'TauA': [(5 + 34./60. + 31.94/60.**2)*15, 22 + 0 + 52.2/60.**2],
    'CasA': [(23 + 23./60. + 24./60.**2)*15, 58 + 48/60. + 54./60.**2],
    'CygA': [(19. + 59/60. + 28.356/60.**2)*15, 40 + 44./60. + 2.097/60.**2],
    'jupiter': None,
    'sun':None,
    'saturn':None,
    'moon':None
}
comap_longitude = -(118 + 16./60. + 56./60.**2)
comap_latitude  =   37.0 + 14./60. + 2/60.**2

def sex2deg(dms,hours=False):

    d,m,s = dms.split(':')
    if float(d) == 0:
        sign = 1
    else:
        sign = float(d)/np.abs(float(d))

    out = np.abs(float(d)) + float(m)/60. + float(s)/60.**2
    out *= sign
    if hours:
        return out*15.
    else:
        return out

def deg2sex(x,hours=False):
    
    if hours:
        x /= 15.
    
    if x == 0:
        sign = 1
    else:
        sign = x/np.abs(x)
    x = np.abs(x)
    
    d = np.floor(x)
    m = np.floor((x -d)*60.)
    s = ((x-d)*60. - m)*60.

    return '{:02d}:{:02d}:{:.2f}'.format(int(sign*d),int(m),float(s))


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

def UnRotate(ra, dec, r0, d0, p0):
    """
    Rotate coordinates to be relative to some ra/dec and sky rotation pang
    
    All inputs in degrees

    """
    skyVec = hp.ang2vec((90.-dec)*np.pi/180., ra*np.pi/180.)

    outVec = RotateR(skyVec, p0)
    outVec = RotateTheta(outVec, d0)
    outVec = RotatePhi(outVec, r0)

    _dec, _ra = hp.vec2ang(outVec)
    _dec = (np.pi/2. - _dec)*180./np.pi
    _ra = _ra * 180./np.pi

    _ra[_ra > 360] -= 360.
    _ra[_ra < 0] += 360.
    return _ra, _dec

def AngularSeperation(phi0,theta0,phi1,theta1, degrees=True):
    """
    phi - longitude parameters
    theta- latitude parameters
    """    
    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    mid = np.sin(theta0*c)*np.sin(theta1*c) + np.cos(theta0*c)*np.cos(theta1*c)*np.cos((phi1-phi0)*c)
    
    return np.arccos(mid)/c


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

def getPlanetPosition(source, lon, lat, mjdtod, allpos=False):
    """
    Get ra, dec and earth-source distance

    args:
    source name (e.g. JUPITER)
    lon : longitude of telescope (degrees)
    lat : latitude of telescope (degrees)
    mjdtod : Modified Julian date (arraylike)
    """

    if 'JUPITER' in source.upper():
        pid = 5
    elif 'SATURN' in source.upper():
        pid = 6
    elif 'MOON' in source.upper():
        pid = 3
    else:
        pid = 0

    r0, d0, jdia = rdplan(mjdtod, pid, lon, lat)
    jdist = planet(mjdtod, pid) # Planet heliocentric position
    edist = planet(mjdtod, 3) # Earth heliocentric position
    rdist = np.sqrt(np.sum((jdist[:3,:] - edist[:3,:])**2,axis=0))

    if allpos:
        return r0, d0, rdist
    else:
        r0 = np.mean(r0)*180./np.pi
        d0 = np.mean(d0)*180./np.pi
        dist = np.mean(rdist)
        return r0, d0, dist

def sourcePosition(src, mjd, lon, lat):
    """
    Get the J2000 RA/Dec position of a source defined in CalibratorList
    """

    skypos = CalibratorList[src]

    if isinstance(skypos,type(None)):
        # we must have Sun/Jupiter
        #_mjd = mjd[::10]
        #_mjd[-1] = mjd[-1]
        time_step = np.abs(mjd[1]-mjd[0])*24*3600.
        target_step = 5.*60. # just check the position every 5 minutes
        if time_step < target_step:
            index_step = int(target_step//time_step)
        else:
            index_step = 1
        r0, d0, dist = getPlanetPosition(src, lon, lat, mjd[::index_step], allpos=True)
        r0 *= 180./np.pi
        d0 *= 180./np.pi
        r0 = np.interp(mjd,mjd[::index_step],r0)
        d0 = np.interp(mjd,mjd[::index_step],d0)
        #r0,d0 = precess(r0, d0, mjd)
    else:
        r0, d0 = mjd*0 + skypos[0], mjd*0 + skypos[1]
        r0, d0 = precess2year(r0,d0,mjd) # Change to epoch of observation

    az, el = e2h(r0,d0,mjd,lon ,lat)
    return az, el, r0, d0

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

    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    ra, dec = pysla.h2e(az*c, el*c, mjd, lon*c, lat*c)
    return ra/c, dec/c

def e2h(ra, dec, mjd, lon, lat, degrees=True, return_lha=False):
    """
    Horizon to equatorial coordinates

    args:
    az - arraylike, azimuth
    el - arraylike, elevation
    mjd- arraylike, modified julian date
    lon- double, longitude
    lat- double, latitude
    """

    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    if not isinstance(ra, np.ndarray):
        ra = np.array([ra])
    if not isinstance(dec, np.ndarray):
        dec = np.array([dec])
    if not isinstance(mjd, np.ndarray):
        mjd = np.array([mjd])

    az, el, lha = pysla.e2h(ra*c, dec*c, mjd, lon*c, lat*c)
    
    if return_lha:
        return az/c, el/c, lha/c
    else:
        return az/c, el/c


def precess(ra, dec, mjd, degrees=True):
    """
    Precess coordinate system to FK5 J2000.
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    """

    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    raout = ra.astype(np.float)*c
    decout = dec.astype(np.float)*c

    pysla.precess(raout,
                  decout, 
                  mjd.astype(np.float))
    return raout/c, decout/c

def prenut(ra, dec, mjd, degrees=True):
    """
    Precess coordinate system to FK5 J2000.
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    """
    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    raout = ra.astype(np.float)*c
    decout = dec.astype(np.float)*c

    pysla.prenut(raout,
                  decout, 
                  mjd.astype(np.float))
    return raout/c, decout/c


def precess2year(ra, dec, mjd, degrees=True):
    """
    Precess coodinrate system to FK5 J2000.
    
    args:
    ra - arraylike, right ascension
    dec- arraylike, declination
    mjd- arraylike
    """

    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    if not isinstance(ra, np.ndarray):
        ra = np.array([ra])
    if not isinstance(dec, np.ndarray):
        dec = np.array([dec])
    if not isinstance(mjd, np.ndarray):
        mjd = np.array([mjd])

    raout = ra.astype(np.float)*c
    decout = dec.astype(np.float)*c

    pysla.precess_year(raout,
                       decout, 
                       mjd.astype(np.float))
    return raout/c, decout/c



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
    
    if degrees:
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
    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    gl, gb = pysla.e2g(ra*c, dec*c)
    return gl/c, gb/c

def g2e(gl, gb, degrees=True):
    """
    Galactic to Equatorial
    
    args:
    gl - arraylike, right ascension
    gb- arraylike, declination
    """
    if degrees:
        c = np.pi/180.
    else:
        c = 1.

    ra, dec = pysla.g2e(gl*c, gb*c)
    return ra/c, dec/c
