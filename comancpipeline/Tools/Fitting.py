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

def DFuncGaussRotLstSq(*args, **kwargs):
    P = args[0]
    func = args[1]
    limits = args[2]
    #z = args[5]
    x, y, z,ra0,dec0 = args[3:]

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

def Gauss2dRot(P, x, y, ra_c, dec_c):
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

def Gauss2dRotPlaneLimits(P):
    A, x0, sigx, y0, sigy, a, B, Gx, Gy = P

    #limits = [(sigx > 10./60./2.355) | (sigx < 0),
    #         (sigy > 10./60./2.355) | (sigy < 0),
    #         (A < 0),
    #         (np.sqrt(x0**2 + y0**2) > 1),
    #         (a < 0) | (a > np.pi)]

    limits = [(A < 0)]
              #(sigx > sigy),
              #(a > np.pi) | (a < 0)]

              
    return any(np.array(limits))




# Error Lstsq
def ErrorLstSq(*args, **kwargs):
    P = args[0]
    func = args[1]
    limits = args[2]
    #z = args[5]
    x, y, z,ra0,dec0 = args[3:]

    if limits(P):
        return 0.*z + 1e32
    else:
        #print(np.sum((z - func(P,x,y,ra0,dec0, **kwargs))**2),flush=True)
        return func(P,x,y,ra0,dec0, **kwargs) - z

def ErrorFmin(*args, **kwargs):
    P = args[0]
    func = args[1]
    limits = args[2]
    #z = args[5]
    x, y, z,ra0,dec0 = args[3:]

    if limits(P):
        return 1e32
    else:
        return np.sum( (z - func(P,x,y,ra0,dec0, **kwargs))**2)
