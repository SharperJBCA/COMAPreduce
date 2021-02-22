# Models for describing different calibrator sources
import numpy as np
from comancpipeline.Tools import Coordinates 
from datetime import datetime
from astropy.time import Time

kb = 1.38064852e-23
cspeed  = 3e-1 # Gm/second

# At 5.2 AU, ref Weiland et al 2011
nujwmap = np.array([22.85, 33.11, 40.82, 60.85, 93.32])
Tjwmap  = np.array([136.2, 147.2, 154.8, 165.6, 173.5])
nujkarim2014 = [34.688,
               34.188,
               33.688,
               33.188,
               32.688,
               32.188,
               31.688,
               31.188,
               30.688,
               30.188,
               29.688,
               29.188,
               28.688,
               28.188,
               27.688]
Tjkarim2014 = [151.013,
               150.385,
               149.615,
               149.876,
               148.534,
               147.707,
               147.235,
               146.635,
               145.72,
               145.347,
               144.794,
               143.911,
               143.225,
               142.369,
               141.757]

jupfit  = np.poly1d(np.polyfit(np.log10(nujkarim2014), np.log10(Tjkarim2014), 1))
jupAng0 = 2.481e-8 # sr

def JupiterFlux(nu, mjd, lon=0, lat=0, source='jupiter',allpos=False,return_jansky=False):
    """
    """

    r0, d0, dist = Coordinates.getPlanetPosition(source, lon, lat, mjd,allpos=allpos)

    jupAng = jupAng0*(5.2/dist)**2
    
    if return_jansky:
        return 2. * kb * (nu/cspeed)**2 * 10**jupfit(np.log10(nu)) * jupAng * 1e26
    else:
        return 10**jupfit(np.log10(nu)) * jupAng , dist

def CygAFlux(nu, mjd=None, lon=0,lat=0,source='CygA',**kwargs):
    """
    Flux of CygA from Baars 1977
    or from Weiland 2011
    """

    a = 7.161
    b = -1.244
    c = 0

    a = 1.480
    b = -1.172

    JyPerK = 2. * kb * (nu/cspeed)**2 * 1e26
    logS = a + b*np.log10(nu/40) + c*np.log10(nu*1e3)**2

    return 10**logS
    

def K2S(nu):
    return 2. * kb * (nu/cspeed)**2 * 1e26

def CasAFlux(nu, mjd=None, lon=0,lat=0,source='CasA',**kwargs):
    """
    Flux of CygA from Baars 1977
    or from Weiland 2011
    """

    a = 7.161
    b = -1.244
    c = 0

    a = 2.204
    b = -0.712
    c = 0.0

    JyPerK = 2. * kb * (nu/cspeed)**2 * 1e26
    logS = a + b*np.log10(nu/40) + c*np.log10(nu/40.)**2
    #print('Jy/K', JyPerK, 10**logS)

    rate = 0.55/100.
    years = (mjd - Time(datetime(2011,1,1),format='datetime').mjd)/365.25
    sec= (1 - rate*years)

    return 10**logS*sec
    
def TauAFlux(nu, mjd=None, lon=0,lat=0,source='TauA',**kwargs):
    """
    Flux of CygA from Baars 1977
    or from Weiland 2011
    """


    a = 2.502
    b = -0.350
    c= 0

    logS = a + b*np.log10(nu/40) + c*np.log10(nu/40.)**2

    rate = 0.21/100.
    years = (mjd - Time(datetime(2011,1,1),format='datetime').mjd)/365.25
    sec= (1 - rate*years)


    return 10**logS*sec
    
