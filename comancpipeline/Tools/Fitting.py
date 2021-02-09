import numpy as np

# FUNCTION FOR FITTING ROUTINES

def Plane(P, x, y):
    
    return P[0]*(x) + P[1]*(y) 

def Gauss2d(P, x, y, ra_c, dec_c, plane=False):
    
    X = (x - ra_c - P[2])
    Y = (y - dec_c - P[3])

    a = (X/P[1])**2
    b = (Y/P[1])**2

    model = P[0] * np.exp( - 0.5 * (a + b)) + P[4]
    if plane:
        model += Plane([P[5], P[6]], x, y)
    return model

def Gauss2dLimits(P):
    return (P[1] > 10./60./2.355) | (P[1] < 0) | (P[0] < 0) | (np.sqrt(P[2]**2 + P[3]**2) > 60./60.)

def DFuncLstSq(*args, **kwargs):
    P = args[0]
    func = args[1]
    limits = args[2]
    #z = args[5]
    x, y, z,ra0,dec0 = args[3:]
    
    m = np.exp(-0.5 * ((x - P[2])**2 + (y -P[3])**2)/P[1]**2)
    r2 = (x - P[2])**2 + (y - P[3])**2
    return np.array([m, 
            r2/P[1]**3 * m,
            P[0] * (x - P[2])/ P[1]**2 * m,
            P[0] * (y - P[3])/ P[1]**2 * m,
            np.ones(m.size)]).T

class Gauss2dRotCondon:
    """
    Based on the document: Erorrs in Elliptical Gaussian Fits, J. J. Condon 1997
    """

    def __init__(self):
        self.__name__ = Gauss2dRot.__name__

    def __call__(self,*args):
        return self.func(*args)

    def func(self,P, xy):
        x,y = xy
        A, x0, sigx, y0, sigy, beta, B = P
        
        Xr = (x - x0)/sigx 
        Yr = (y - y0)/sigy

        model = A * np.exp( - 0.5 * (Xr**2 + Yr**2) - beta*Xr*Yr) + B
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



class Gauss2dRot:

    def __init__(self):
        self.__name__ = Gauss2dRot.__name__

    def __call__(self,*args,**kwargs):
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

class Gauss2dRot_FixedPos:

    def __init__(self):
        self.__name__ = Gauss2dRot_FixedPos.__name__

    def __call__(self,*args,**kwargs):
        A, sigx, sigy, B = args[0]
        if (sigx <= 0) | (sigy <= 0):
            return np.inf
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



def DFuncGaussRotLstSq(*args, **kwargs):
    P = args[0]
    func = args[1][0]
    limits = args[1][1]
    #z = args[5]
    x, y, z,ra0,dec0 = args[1][2:]

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

    return np.array([m, 
                     A * m * (Cx * (x - x0) + 0.5*(y-y0) * Cr) , # offset x
                     A * m / sigx**3 * Zx,  # sigma x
                     A * m * (Cy * (y - y0) + 0.5*(x-x0) * Cr) , # offset y
                     A * m / sigy**3 * Zy, # sigma y
                     A * m * Rc, # rotation angle
                     np.ones(m.size)]).T

def Gauss2dRot_(P, x, y, ra_c, dec_c):
    A, x0, sigx, y0, sigy, phi, B = P
    Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
    Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

    model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B
    return model

def Gauss2dRotLimits(P):
    A, x0, sigx, y0, sigy, a, B = P

    limits = [(sigx > 10./60./2.355) | (sigx < 0),
              (sigy > 10./60./2.355) | (sigy < 0),
              (A < 0),
              (np.sqrt(x0**2 + y0**2) > 1),
              (a < 0) | (a > np.pi)]


              
    return any(np.array(limits))

# WITH GRADIENT
def DFuncGaussRotPlaneLstSq(*args, **kwargs):
    P = args[0]
    func = args[1]
    limits = args[2]
    #z = args[5]
    x, y, z,ra0,dec0 = args[3:]

    A, x0, sigx, y0, sigy, phi, B, Gx, Gy = P
    
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

    return np.array([m, 
                     A * m * (Cx * (x - x0) + 0.5*(y-y0) * Cr) , # offset x
                     A * m / sigx**3 * Zx,  # sigma x
                     A * m * (Cy * (y - y0) + 0.5*(x-x0) * Cr) , # offset y
                     A * m / sigy**3 * Zy, # sigma y
                     A * m * Rc, # rotation angle
                     np.ones(m.size),
                     x,
                     y]).T

def Gauss2dRotPlane(P, x, y, ra_c, dec_c):
    A, x0, sigx, y0, sigy, phi, B, Gx, Gy = P
    Xr = (x - x0)/sigx * np.cos(phi) + (y-y0)/sigx * np.sin(phi)
    Yr =-(x - x0)/sigy * np.sin(phi) + (y-y0)/sigy * np.cos(phi)

    model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B + Gx*(x-x0) + Gy*(y-y0)
    return model

def Gauss2dSymmetricPlane(P, x, y, ra_c, dec_c):
    A, x0, y0, sig, B, Gx, Gy = P
    Xr = (x - x0)/sig 
    Yr = (y - y0)/sig

    model = A * np.exp( - 0.5 * (Xr**2 + Yr**2)) + B + Gx*(x-x0) + Gy*(y-y0)
    return model


def Gauss2dRotPlaneLimits(P):
    A, x0, sigx, y0, sigy, a, B, Gx, Gy = P

    limits = [(A < 0)]
              
    return any(np.array(limits))

def Gauss2dSymmetricPlaneLimits(P):
    A, x0, y0, sig, B, Gx, Gy = P

    limits = [(A < 0) , (sig < 0)]
              
    return any(np.array(limits))





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

    if (P[-2] < -np.pi/2.) | (P[-2] > np.pi/2.):
        return -1e32


    if limits(P):
        return -1e32
    else:
        #print(np.sum((z - func(P,x,y,ra0,dec0, **kwargs))**2),flush=True)
        return -np.sum( (func(P,xy, **otherkeys) - z)**2/cov )


def ErrorFmin(*args, **kwargs):
    print(args)
    P = args[0]
    func = kwargs['args'][0]
    limits = kwargs['args'][1]
    #z = args[5]
    x, y, z,ra0,dec0 = kwargs['args'][2:]

    if limits(P):
        return 1e32
    else:
        return np.sum( (z - func(P,x,y,ra0,dec0, **kwargs))**2)
