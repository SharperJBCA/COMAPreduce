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
