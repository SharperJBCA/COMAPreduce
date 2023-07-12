# Models for describing different calibrator sources
import numpy as np
from comancpipeline.Tools import Coordinates 
from datetime import datetime
from astropy.time import Time

kb = 1.38064852e-23
cspeed  = 3e-1 # Gm/second
cspeed_full = 299792458.
# At 5.2 AU, ref Weiland et al 2011
nujwmap = np.array([22.85, 33.11, 40.82, 60.85, 93.32])
Tjwmap  = np.array([136.2, 147.2, 154.8, 165.6, 173.5])
nujkarim2014 = np.array([34.688,
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
                         27.688])
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

jupfit  = np.poly1d(np.polyfit(np.log10(nujkarim2014/30.), np.log10(Tjkarim2014), 1))
jupAng0 = 2.481e-8 # sr

def JupiterFlux(nu, mjd, lon=0, lat=0, source='jupiter',allpos=False,return_jansky=False):
    """
    """

    r0, d0, dist = Coordinates.getPlanetPosition(source, lon, lat, mjd,allpos=allpos)

    jupAng = jupAng0*(5.2/dist)**2
    
    if return_jansky:
        return 2. * kb * (nu/cspeed)**2 * 10**jupfit(np.log10(nu/30.)) * jupAng * 1e26
    else:
        return 10**jupfit(np.log10(nu/30.)) * jupAng , dist

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

    if len(mjd) > 0:
        return 10**logS * np.ones(len(mjd))
    else:
        return 10**logS
    

def K2S(nu):
    return 2. * kb * (nu/cspeed)**2 * 1e26

def CasASecular(nu,mjd):
    rate = 0.55/100.
    years = (mjd - Time(datetime(2011,1,1),format='datetime').mjd)/365.25
    sec= (1 - rate*years)

    return sec

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

    sec= CasASecular(nu,mjd) 

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
    
class JupiterFluxModel:
    def __init__(self,flux_model='karim2014'):
        
        self.flux_model_name = flux_model

        self.model = getattr(self,self.flux_model_name)
        self.jupAng0 = 2.481e-8 # sr
        self.ref_year = None
        # Setup parameters
        self.model(0)

    def __call__(self, nu, mjd, **kwargs):
        """
        """

        self.model = getattr(self,self.flux_model_name)

        F = self.model(nu,mjd, allpos=True, return_jansky=True, **kwargs)
        return F

    def karim2014(self,nu,mjd=51544, lon=0, lat=0, source='jupiter',allpos=False,return_jansky=False,**kwargs):
        """
        args:
        nu - in GHz
        mjd - 
        """
        
        self.P = {'a':jupfit[0],
                  'b':jupfit[1],
                  'c': 0}

        r0, d0, dist = Coordinates.getPlanetPosition(source, lon, lat, mjd,allpos=allpos)
        jupAng = self.jupAng0*(5.2/dist)**2
    
        if return_jansky:
            return 2. * kb * (nu/cspeed)**2 * 10**jupfit(np.log10(nu/30.)) * jupAng * 1e26
        else:
            return 10**jupfit(np.log10(nu/30.)) * jupAng , dist

        

class TauAFluxModel:
    def __init__(self,flux_model='weiland_combined',secular_model='weiland_secular'):
        
        self.flux_model_name = flux_model
        self.sec_model_name = secular_model

        self.model = getattr(self,self.flux_model_name)
        self.sec_model = getattr(self,self.sec_model_name)

        # Setup parameters
        self.model(0)
        self.sec_model(0,0)


    def __call__(self, nu, mjd, **kwargs):
        """
        """

        self.model = getattr(self,self.flux_model_name)
        self.sec_model = getattr(self,self.sec_model_name)


        F = self.model(nu,**kwargs)
        return F*self.sec_model(nu,mjd,**kwargs)

    def weiland_secular(self,nu,mjd,**kwargs):
        """
        Weiland 2011, Table 16
        """
        self.rate = 0.21/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec

    def hafez_secular(self,nu,mjd,**kwargs):
        """
        Hafez 2008
        """
        self.rate = 0.22/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec

    def trotter_secular(self,nu,mjd,**kwargs):
        """
        Trotter 2020
        """
        self.rate = 0.1/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec

    def baars(self,nu, **kwargs):
        """
        Baars 1977, Table 3
        """

        self.P = {'a':3.915,
                  'b':-0.299,
                  'c':0}
        self.Perr = {'a':0.031,
                     'b':0.009,
                     'c':0}

        self.ref_year = 1970

        model = self.P['a'] + self.P['b']*np.log10(nu*1e3)

        return 10**model

    def weiland_combined(self,nu,**kwargs):
        """
        Weiland 2011, Table 17
        """

        self.P = {'a':2.506,
                  'b':-0.302,
                  'c':0}
        self.Perr = {'a':0.003,
                     'b':0.005,
                     'c':0}

        self.ref_year = 2005

        model = self.P['a'] + self.P['b']*np.log10(nu/40.)

        return 10**model

    def weiland_wmaponly(self,nu,**kwargs):
        """
        Weiland 2011, Table 17
        """

        self.P = {'a':2.502,
                  'b':-0.350,
                  'c':0}
        self.Perr = {'a':0.005,
                     'b':0.026,
                     'c':0}

        self.ref_year = 2005

        model = self.P['a'] + self.P['b']*np.log10(nu/40.)

        return 10**model

    def hafez(self,nu,**kwargs):
        """
        Hafez 2008 - 33GHz reference data
        """

        self.P = {'a':np.log10(332.8),
                  'b':-0.32,
                  'c':0}

        self.ref_year = 2001

        model = self.P['a'] + self.P['b']*np.log10(nu/33.)

        return 10**model

class CasAFluxModel:
    def __init__(self,flux_model='weiland_combined',secular_model='weiland_secular'):
        
        self.flux_model_name = flux_model    
        self.sec_model_name = secular_model

        self.model = getattr(self,self.flux_model_name)
        self.sec_model = getattr(self,self.sec_model_name)

        # Setup parameters
        self.model(0)
        self.sec_model(0,0)

    def __call__(self, nu, mjd, **kwargs):
        """
        """

        self.model = getattr(self,self.flux_model_name)
        self.sec_model = getattr(self,self.sec_model_name)


        F = self.model(nu,**kwargs)
        return F*self.sec_model(nu,mjd,**kwargs)

    def weiland_combined(self,nu,**kwargs):
        """
        Weiland 2011, Table 17
        """
        
        self.P = {'a':2.204,
                  'b':-0.682,
                  'c':0.038}
        self.Perr = {'a':0.002,
                     'b':0.011,
                     'c':0.008}

        self.ref_year = 2000

        model = self.P['a'] + self.P['b']*np.log10(nu/40.) + self.P['c']*np.log10(nu/40.)**2

        return 10**model

    def weiland_wmaponly(self,nu,**kwargs):
        """
        Weiland 2011, Table 17
        """
        
        self.P = {'a':2.204,
                  'b':-0.712,
                  'c':0.}
        self.Perr = {'a':0.002,
                     'b':0.018,
                     'c':0.}

        self.ref_year = 2000

        model = self.P['a'] + self.P['b']*np.log10(nu/40.) + self.P['c']*np.log10(nu/40.)**2

        return 10**model

    def weiland_secular(self,nu,mjd,**kwargs):
        """
        Weiland 2011, Table 16
        """
        self.rate = 0.51/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec

    def hafez(self,nu,**kwargs):
        """
        Hafez 2008, 33GHz
        """
        
        self.P = {'a':np.log10(182),
                  'b':-0.69,
                  'c':0.}
        self.Perr = {'a':0.002,
                     'b':0.02,
                     'c':0.}

        self.ref_year = 2000

        model = self.P['a'] + self.P['b']*np.log10(nu/33.) + self.P['c']*np.log10(nu/33.)**2

        return 10**model

    def hafez_secular(self,nu,mjd,**kwargs):
        """
        Hafez 2008
        """
        self.rate = 0.394/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec

    def baars(self,nu, **kwargs):
        """
        Baars 1977, Table 3
        """

        self.P = {'a':5.880,
                  'b':-0.792,
                  'c':0}
        self.Perr = {'a':0.025,
                     'b':0.007,
                     'c':0}

        self.ref_year = 1965

        model = self.P['a'] + self.P['b']*np.log10(nu*1e3)

        return 10**model
    def baars_secular(self,nu,mjd, **kwargs):
        """
        Baars 1977, in text
        """
        self.rate = (0.394 - 0.3 * np.log10(nu))/100.
        years = (mjd - Time(datetime(self.ref_year,1,1),format='datetime').mjd)/365.25
        sec= (1 - self.rate*years)
        
        return sec


