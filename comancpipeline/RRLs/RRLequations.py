import numpy as np

def gaunt(Te,v, Z = 1):
    """
    Gaunt factor from Planck Int. XV 2014

    Line gaunt factor is different frm the Dickinson 2003 version?
    """

    T4 = Te/1e4
    A = 5.960 - np.sqrt(3)/np.pi * np.log(Z * v * T4**-1.5)

    return np.log(np.exp(A) + 2.71828)

def gaunt2(Te,v):

    return np.log(4.955e-2 / v) + 1.5*np.log(Te)

def altenhoff(Te,v):
    
    A =  np.log(4.955e-2 / v) + 1.5*np.log(Te) 
    return 0.366*v**0.1 * Te**-0.15 * A

def gaunt3(Te,v):
    x = [0.4 ,1.4 ,2.3 ,10. ,30. ,44  ,70  ,100]
    y = [6.39,5.73,5.47,4.70,4.12,3.92,3.67,3.49]
    return np.interp(v,x,y)

def line_ratio(Te,v):
    return 6.985e3 * Te**-1.15 * v**1.1 / 1.08 / altenhoff(Te,v)

def line_ratio_mdl2(ratio,v):
    """
    From Balser 2011, Quireza 2006 APJ 653
    """

    return (7103.3 * v**1.1 / ratio / 1.08)**0.87
 
def line_ratio_mdl3(ratio,v,dv,Te=7000.):
    return (6.985e3/gaunt(Te,v) * v**1.1 / ratio / dv / 1.08)**0.87
