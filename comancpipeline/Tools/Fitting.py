import numpy as np
from scipy.optimize import minimize
import emcee
from comancpipeline.Tools import stats
from tqdm import tqdm
# FUNCTION FOR FITTING ROUTINES

class Gauss2dRot:

    def __init__(self):
        self.__name__ = Gauss2dRot.__name__

    def __call__(self,*args,**kwargs):
        A, x0, sigx, y0, sigy, phi, B = args[0]
        if (sigx <= 0) | (sigy <= 0):
            return np.inf
        if (phi <= -2*np.pi) | (phi >=2*np.pi):
            return np.inf
        return self.func(*args,**kwargs)

    def func(self,P, xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B = P
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)
        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B
        return model

    def deriv(self,P,xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B = P
        
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        r2 = (Xr**2 + Yr**2)
        m = np.exp(-0.5 * r2)

        # Constants for offset derivatives
        Cx  = (np.cos(phi)**2/sigx**2 + np.sin(phi)**2/sigy**2)
        Cy  = (np.cos(phi)**2/sigy**2 + np.sin(phi)**2/sigx**2)
        Cr  = (np.sin(2*phi) /sigx**2 - np.sin(2*phi) /sigy**2)

        # Constants for sigma derivatives
        Zx = (x-x0)**2 * np.cos(phi)**2 + (y-y0)**2 * np.sin(phi)**2 + (x-x0) * (y-y0) * np.sin(2*phi)
        Zy = (x-x0)**2 * np.sin(phi)**2 + (y-y0)**2 * np.cos(phi)**2 - (x-x0) * (y-y0) * np.sin(2*phi) 

        # Constant for rotation derivative(x-x0)
        Rc = 0.5 * ((x-x0)**2-(y-y0)**2) * ( np.sin(2*phi)/sigx**2 - np.sin(2*phi)/sigy**2) -  (x-x0) * (y-y0) * np.cos(2*phi) * (1./sigx**2 - 1./sigy**2)

        output= np.array([m, 
                          A * m * (Cx * (x - x0) + 0.5*(y-y0) * Cr) , # offset x
                          A * m / sigx**3 * Zx,  # sigma x
                          A * m * (Cy * (y - y0) + 0.5*(x-x0) * Cr) , # offset y
                          A * m / sigy**3 * Zy, # sigma y
                          A * m * Rc, # rotation angle
                          np.ones(m.size)])

        return np.transpose(output)

class Gauss2dRot_Gradient:

    def __init__(self):
        self.__name__ = Gauss2dRot_Gradient.__name__

    def __call__(self,*args,**kwargs):
        A, x0, sigx, y0, sigy, phi, B, Gx, Gy, Gxy = args[0]

        return self.func(*args,**kwargs)

    def func(self,P, xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B, Gx, Gy, Gxy = P
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)
        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B + Gx*(x-x0) + Gy*(y-y0) + Gxy*(x-x0)*(y-y0)
        return model

    def deriv(self,P,xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B = P
        
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        r2 = (Xr**2 + Yr**2)
        m = np.exp(-0.5 * r2)

        # Constants for offset derivatives
        Cx  = (np.cos(phi)**2/sigx**2 + np.sin(phi)**2/sigy**2)
        Cy  = (np.cos(phi)**2/sigy**2 + np.sin(phi)**2/sigx**2)
        Cr  = (np.sin(2*phi) /sigx**2 - np.sin(2*phi) /sigy**2)

        # Constants for sigma derivatives
        Zx = (x-x0)**2 * np.cos(phi)**2 + (y-y0)**2 * np.sin(phi)**2 + (x-x0) * (y-y0) * np.sin(2*phi)
        Zy = (x-x0)**2 * np.sin(phi)**2 + (y-y0)**2 * np.cos(phi)**2 - (x-x0) * (y-y0) * np.sin(2*phi) 

        # Constant for rotation derivative(x-x0)
        Rc = 0.5 * ((x-x0)**2-(y-y0)**2) * ( np.sin(2*phi)/sigx**2 - np.sin(2*phi)/sigy**2) -  (x-x0) * (y-y0) * np.cos(2*phi) * (1./sigx**2 - 1./sigy**2)

        output= np.array([m, 
                          A * m * (Cx * (x - x0) + 0.5*(y-y0) * Cr) , # offset x
                          A * m / sigx**3 * Zx,  # sigma x
                          A * m * (Cy * (y - y0) + 0.5*(x-x0) * Cr) , # offset y
                          A * m / sigy**3 * Zy, # sigma y
                          A * m * Rc, # rotation angle
                          np.ones(m.size)])

        return np.transpose(output)


class Gauss2dRot_Gradient2:

    def __init__(self):
        self.__name__ = Gauss2dRot_Gradient.__name__

    def __call__(self,*args,**kwargs):
        A, x0, sigx, y0, sigy, phi, B, Gx, Gy, Gxy,Gx2,Gy2,Gxy2,Gx2y,Gx2y2 = args[0]

        return self.func(*args,**kwargs)

    def func(self,P, xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B, Gx, Gy, Gxy,Gx2,Gy2,Gxy2,Gx2y,Gx2y2 = P
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        poly = Gx*(x-x0) + Gy*(y-y0) + Gxy*(x-x0)*(y-y0) + Gx2*(x-x0)**2 + Gy2*(x-x0)**2 + Gxy2*(x-x0)*(y-y0)**2 + Gx2y*(x-x0)**2*(y-y0) + Gx2y2*(x-x0)**2*(y-y0)**2
        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B + poly
        return model

    def deriv(self,P,xy):
        x,y = xy
        A, x0, sigx, y0, sigy, phi, B = P
        
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        r2 = (Xr**2 + Yr**2)
        m = np.exp(-0.5 * r2)

        # Constants for offset derivatives
        Cx  = (np.cos(phi)**2/sigx**2 + np.sin(phi)**2/sigy**2)
        Cy  = (np.cos(phi)**2/sigy**2 + np.sin(phi)**2/sigx**2)
        Cr  = (np.sin(2*phi) /sigx**2 - np.sin(2*phi) /sigy**2)

        # Constants for sigma derivatives
        Zx = (x-x0)**2 * np.cos(phi)**2 + (y-y0)**2 * np.sin(phi)**2 + (x-x0) * (y-y0) * np.sin(2*phi)
        Zy = (x-x0)**2 * np.sin(phi)**2 + (y-y0)**2 * np.cos(phi)**2 - (x-x0) * (y-y0) * np.sin(2*phi) 

        # Constant for rotation derivative(x-x0)
        Rc = 0.5 * ((x-x0)**2-(y-y0)**2) * ( np.sin(2*phi)/sigx**2 - np.sin(2*phi)/sigy**2) -  (x-x0) * (y-y0) * np.cos(2*phi) * (1./sigx**2 - 1./sigy**2)

        output= np.array([m, 
                          A * m * (Cx * (x - x0) + 0.5*(y-y0) * Cr) , # offset x
                          A * m / sigx**3 * Zx,  # sigma x
                          A * m * (Cy * (y - y0) + 0.5*(x-x0) * Cr) , # offset y
                          A * m / sigy**3 * Zy, # sigma y
                          A * m * Rc, # rotation angle
                          np.ones(m.size)])

        return np.transpose(output)




class Gauss2dRot_FixedPos:

    def __init__(self):
        self.__name__ = Gauss2dRot_FixedPos.__name__

    def __call__(self,*args,**kwargs):
        A, sigx, sigy, B = args[0]
        if (sigx <= 0) | (sigy <= 0):
            return 1e32
        return self.func(*args,**kwargs)

    def func(self,P, xy, x0=0, y0=0, phi=0):
        x,y = xy
        A, sigx, sigy, B = P
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)
        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B
        return model

    def deriv(self,P,xy, x0=0, y0=0, phi=0):
        x,y = xy
        A, sigx, sigy, B = P
        
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        r2 = (Xr**2 + Yr**2)
        m = np.exp(-0.5 * r2)

        # Constants for offset derivatives
        Cx  = (np.cos(phi)**2/sigx**2 + np.sin(phi)**2/sigy**2)
        Cy  = (np.cos(phi)**2/sigy**2 + np.sin(phi)**2/sigx**2)
        Cr  = (np.sin(2*phi) /sigx**2 - np.sin(2*phi) /sigy**2)

        # Constants for sigma derivatives
        Zx = (x-x0)**2 * np.cos(phi)**2 + (y-y0)**2 * np.sin(phi)**2 + (x-x0) * (y-y0) * np.sin(2*phi)
        Zy = (x-x0)**2 * np.sin(phi)**2 + (y-y0)**2 * np.cos(phi)**2 - (x-x0) * (y-y0) * np.sin(2*phi) 

        # Constant for rotation derivative(x-x0)
        Rc = 0.5 * ((x-x0)**2-(y-y0)**2) * ( np.sin(2*phi)/sigx**2 - np.sin(2*phi)/sigy**2) -  (x-x0) * (y-y0) * np.cos(2*phi) * (1./sigx**2 - 1./sigy**2)

        output= np.array([m, 
                          A * m / sigx**3 * Zx,  # sigma x
                          A * m / sigy**3 * Zy, # sigma y
                          np.ones(m.size)])

        return np.transpose(output)

    def auto_comps(self,f,y,d1,d2):
        return (0.5*np.sum(d1**2) + np.sum(f*d2) - np.sum(y*d2))

    def cross_comps(self,f,y,d1a,d1b,d2):
        return (0.5*np.sum(d1a*d1b) + np.sum(f*d2) - np.sum(y*d2))

    def covariance(self,P,xy,z,e, x0=0, y0=0, phi=0):
        """
        
        """
        x,y = xy
        A, sigx, sigy, B = P

        X = (x - x0) * np.cos(phi) + (y-y0) * np.sin(phi)
        Y =-(x - x0) * np.sin(phi) + (y-y0) * np.cos(phi)
        

        deriv = {'dA'    : self.func(P,xy,x0,y0,phi)/e,
                 'dA2'   : 0.,
                 'dB'    : np.ones(x.size)/e,
                 'dB2'   : 0.,
                 'dSigX' : self.func(P,xy,x0,y0,phi)*X**2/sigx**3/e,
                 'dSigY' : self.func(P,xy,x0,y0,phi)*Y**2/sigy**3/e}
        deriv['dSigX2'] = deriv['dSigX']*(X**2/sigx**3 - 3./sigx)
        deriv['dSigY2'] = deriv['dSigY']*(Y**2/sigy**3 - 3./sigy)
        deriv['dSigXSigY'] = deriv['dSigX']*deriv['dSigY']/self.func(P,xy,x0,y0,phi)*e # to cancel the double instance of the uncertainty
        deriv['dASigX'] = deriv['dSigX']/A
        deriv['dASigY'] = deriv['dSigY']/A

        c    = {'00':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dA2'])   , # AUTO
                '10':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dSigX']  , deriv['dASigX']),
                '30':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dA']     , 0),
                '31':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dSigX']  , 0),
                '32':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dSigY']  , 0),
                '11':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dSigX'], deriv['dSigX2']), # AUTO
                '20':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dSigY']  , deriv['dASigY']),
                '21':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dSigX'], deriv['dSigY']  , deriv['dSigXSigY']),
                '22':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dSigY'], deriv['dSigY2']), # AUTO
                '33':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dB2'])} # AUTO

        V = np.array([[c['00'],c['10'],c['20'],c['30']],
                      [c['10'],c['11'],c['21'],c['31']],
                      [c['20'],c['21'],c['22'],c['32']],
                      [c['30'],c['31'],c['32'],c['33']]])

        C = np.linalg.inv(V)
        
        return np.sqrt(np.diag(C))


class Gauss2dRot_Gradient_FixedPos:

    def __init__(self):
        self.__name__ = Gauss2dRot_Gradient_FixedPos.__name__

    def __call__(self,*args,**kwargs):
        A, sigx, sigy, B, Gx, Gy, Gxy = args[0]
        if (sigx <= 0) | (sigy <= 0):
            return np.inf
        
        return self.func(*args,**kwargs)

    def func(self,P, xy, x0=0, y0=0, phi=0):
        x,y = xy
        A, sigx, sigy, B, Gx, Gy, Gxy = P
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)
        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B + Gx*(x-x0) + Gy*(y-y0) + Gxy*(x-x0)*(y-y0)
        return model

    def deriv(self,P,xy, x0=0, y0=0, phi=0):
        x,y = xy
        A, sigx, sigy, B = P
        
        Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
        Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

        r2 = (Xr**2 + Yr**2)
        m = np.exp(-0.5 * r2)

        # Constants for offset derivatives
        Cx  = (np.cos(phi)**2/sigx**2 + np.sin(phi)**2/sigy**2)
        Cy  = (np.cos(phi)**2/sigy**2 + np.sin(phi)**2/sigx**2)
        Cr  = (np.sin(2*phi) /sigx**2 - np.sin(2*phi) /sigy**2)

        # Constants for sigma derivatives
        Zx = (x-x0)**2 * np.cos(phi)**2 + (y-y0)**2 * np.sin(phi)**2 + (x-x0) * (y-y0) * np.sin(2*phi)
        Zy = (x-x0)**2 * np.sin(phi)**2 + (y-y0)**2 * np.cos(phi)**2 - (x-x0) * (y-y0) * np.sin(2*phi) 

        # Constant for rotation derivative(x-x0)
        Rc = 0.5 * ((x-x0)**2-(y-y0)**2) * ( np.sin(2*phi)/sigx**2 - np.sin(2*phi)/sigy**2) -  (x-x0) * (y-y0) * np.cos(2*phi) * (1./sigx**2 - 1./sigy**2)

        output= np.array([m, 
                          A * m / sigx**3 * Zx,  # sigma x
                          A * m / sigy**3 * Zy, # sigma y
                          np.ones(m.size)])

        return np.transpose(output)

    def auto_comps(self,f,y,d1,d2):
        return (0.5*np.sum(d1**2) + np.sum(f*d2) - np.sum(y*d2))

    def cross_comps(self,f,y,d1a,d1b,d2):
        return (0.5*np.sum(d1a*d1b) + np.sum(f*d2) - np.sum(y*d2))

    def covariance(self,P,xy,z,e, x0=0, y0=0, phi=0):
        """
        
        """
        x,y = xy
        A, sigx, sigy, B = P

        X = (x - x0) * np.cos(phi) + (y-y0) * np.sin(phi)
        Y =-(x - x0) * np.sin(phi) + (y-y0) * np.cos(phi)
        

        deriv = {'dA'    : self.func(P,xy,x0,y0,phi)/e,
                 'dA2'   : 0.,
                 'dB'    : np.ones(x.size)/e,
                 'dB2'   : 0.,
                 'dSigX' : self.func(P,xy,x0,y0,phi)*X**2/sigx**3/e,
                 'dSigY' : self.func(P,xy,x0,y0,phi)*Y**2/sigy**3/e}
        deriv['dSigX2'] = deriv['dSigX']*(X**2/sigx**3 - 3./sigx)
        deriv['dSigY2'] = deriv['dSigY']*(Y**2/sigy**3 - 3./sigy)
        deriv['dSigXSigY'] = deriv['dSigX']*deriv['dSigY']/self.func(P,xy,x0,y0,phi)*e # to cancel the double instance of the uncertainty
        deriv['dASigX'] = deriv['dSigX']/A
        deriv['dASigY'] = deriv['dSigY']/A

        c    = {'00':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dA2'])   , # AUTO
                '10':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dSigX']  , deriv['dASigX']),
                '30':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dA']     , 0),
                '31':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dSigX']  , 0),
                '32':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dSigY']  , 0),
                '11':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dSigX'], deriv['dSigX2']), # AUTO
                '20':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dA']   , deriv['dSigY']  , deriv['dASigY']),
                '21':self.cross_comps(self.func(P,xy,x0,y0,phi), z, deriv['dSigX'], deriv['dSigY']  , deriv['dSigXSigY']),
                '22':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dSigY'], deriv['dSigY2']), # AUTO
                '33':self.auto_comps (self.func(P,xy,x0,y0,phi), z, deriv['dB']   , deriv['dB2'])} # AUTO

        V = np.array([[c['00'],c['10'],c['20'],c['30']],
                      [c['10'],c['11'],c['21'],c['31']],
                      [c['20'],c['21'],c['22'],c['32']],
                      [c['30'],c['31'],c['32'],c['33']]])

        C = np.linalg.inv(V)
        
        return np.sqrt(np.diag(C))

class Gauss2dRot_General:

    def __init__(self,defaults={'x0':0,'y0':0,'sigx':0,'sigy_scale':1,'phi':0,'A':0,'B':0},
                 fixed=[],
                 use_bootstrap=False):
        self.__name__ = Gauss2dRot_General.__name__
        
        self.use_bootstrap = use_bootstrap
        fixed = {k:True for k in fixed}
        self.set_fixed(**fixed)
        self.defaults = defaults

        self.param_names = ['A','x0','sigx','y0','sigy_scale','phi','B']
        self.A = 0
        self.B = 0
        self.sigx = 0
        self.sigy_scale = 1
        self.x0 = 0
        self.y0 = 0
        self.phi = 0

        self.idx = {k:i for i,k in enumerate(self.param_names)}

    def limfunc(self,P):

        lims = {'A':lambda P: False,
                'B':lambda P: False,
                'x0':lambda P: False,
                'y0':lambda P: False,
                'phi':lambda P: (P['phi'] < -np.pi/2.) | (P['phi'] > np.pi/2.),
                'sigx': lambda P: (P['sigx'] < 0),
                'sigy_scale': lambda P: (P['sigy_scale'] < 1) | (P['sigy_scale'] > 10)}

        params = self.get_param_names()

        Pz = {k:p for k,p in zip(params,P)}
        lims = [lims[k](Pz) for k in params]

        return any(lims)

    def get_param_names(self):
        
        return [p for p in self.param_names if not self.fixed[p]]

    def __call__(self,P0_dict,xy,z,covariance,P0_priors={},
                 limfunc=None, nwalkers=32, samples=5000, discard=100,thin=15,return_array=False):


        
        self.P0_priors = P0_priors
        if isinstance(limfunc,type(None)):
            self.limfunc = self.limfunc
        else:
            self.limfunc = limfunc

        P0 = [v for k,v in P0_dict.items()]
        self.idx= {k:i for i, k in enumerate(P0_dict.keys())}

        if self.use_bootstrap:
            self.niter = 100
            results = np.zeros((self.niter,self.nparams))
            for i in tqdm(range(self.niter)):
                sel = np.random.uniform(low=0,high=z.size,size=z.size).astype(int)
                results[i] = minimize(self.minimize_errfunc,P0,args=((xy[0][sel],xy[1][sel]),z[sel],covariance[sel]),method='CG').x
            error = stats.MAD(results,axis=0)
            result  = np.nanmedian(results,axis=0)
            
        else:
            # Perform the least-sqaures fit
            result = minimize(self.minimize_errfunc,P0,args=(xy,z,covariance),method='CG')
            pos = result.x + 1e-4 * np.random.normal(size=(nwalkers, len(result.x)))
            sampler = emcee.EnsembleSampler(nwalkers,len(result.x),self.emcee_errfunc, 
                                            args=(xy,z,covariance))
            sampler.run_mcmc(pos,samples,progress=True)
            
            flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
            result = np.nanmedian(flat_samples,axis=0)
            error  = stats.MAD(flat_samples ,axis=0)

            min_chi2 = self.emcee_errfunc(result,xy,z,covariance)
            ddof = len(z)

        Value_dict = {k:result[i] for k, i in self.idx.items()}
        Error_dict = {k:error[i]  for k, i in self.idx.items()}
        if return_array:
            return result, error, flat_samples, min_chi2, ddof
        else:
            return Value_dict, Error_dict, flat_samples, min_chi2, ddof

    def Priors(self,P):
        prior = 0
        for k,v in self.P0_priors.items():
            prior += (P[self.idx[k]]-self.P0_priors[k]['mean'])**2/self.P0_priors[k]['width']**2
        return prior

    def set_fixed(self,**kwargs):
        self.fixed = {'x0':False,'y0':False,'sigx':False,'sigy_scale':False,'phi':False,'A':False,'B':False}
        for k,v in kwargs.items():
            if not k in self.fixed:
                raise KeyError('Key not in self.fixed')
            self.fixed[k] = v
        self.nparams = 0
        for k,v in self.fixed.items():
            if not v:
                self.nparams += 1 
        assert self.nparams > 0, 'All parameters fixed?'


    def set_defaults(self,**kwargs):
        for k,v in kwargs.items():
            self.defaults[k] = v

    def emcee_errfunc(self,P,xy,z,cov):  
        if self.limfunc(P):
            return -1e32
        else:
            chi2 = -np.sum( (self.func(P,xy) - z)**2/cov ) - self.Priors(P)

            if np.isfinite(chi2):
                return chi2
            else:
                print('Inf found:', chi2, P)
                retrun -1e32

    def minimize_errfunc(self,P,xy,z,cov):
        if self.limfunc(P):
            return 1e32
        else:
            return np.sum( (self.func(P,xy) - z)**2/cov )

    def func(self,P, xy):
        x,y = xy
        for parameter in self.param_names:
            if parameter in self.idx:
                self.__dict__[parameter] = P[self.idx[parameter]]
            else:
                self.__dict__[parameter] = self.defaults[parameter]

        if self.fixed['sigy_scale']:
            sigy = self.sigx
        else:
            sigy = self.sigx*self.sigy_scale

        Xr = (x - self.x0)/self.sigx * np.cos(self.phi) + (y-self.y0)/self.sigx * np.sin(self.phi)
        Yr =-(x - self.x0)/sigy * np.sin(self.phi) + (y-self.y0)/sigy * np.cos(self.phi)
        model = self.A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + self.B
        return model


# Error Lstsq
def ErrorLstSq(*args):
    P = args[0]
    func = args[1][0]
    limits = args[1][1]
    #z = args[5]
    xy, z,cov, otherkeys = args[1][2:]
    if limits(P):
        return 0.*z + 1e32
    else:
        return np.sum( (func(P,xy, **otherkeys) - z)**2/cov )

def MC_ErrorLstSq(P,*args):
    #P = args
    func = args[0]
    limits = args[1]
    #z = args[5]
    xy, z,cov, otherkeys = args[2:]

    if limits(P):
        return -1e32
    else:
        #print(np.sum((z - func(P,x,y,ra0,dec0, **kwargs))**2),flush=True)
        return -np.sum( (func(P,xy, **otherkeys) - z)**2/cov )

