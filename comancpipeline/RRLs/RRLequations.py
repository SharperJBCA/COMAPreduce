import numpy as np

def gaunt(Te,v, Z = 1):
    """
    Gaunt factor from Planck Int. XV 2014

    Line gaunt factor is different frm the Dickinson 2003 version?
    """

    T4 = Te/1e4
    A = 5.960 - np.sqrt(3)/np.pi * np.log(Z * v * T4**-1.5)

    return np.log(np.exp(A) + 2.71828)

def line_ratio(Te,v):
    return 6.985e3 * Te**-1.15 * v**1.1 / 1.08 #/ gaunt(Te,v)

def line_ratio_mdl2(ratio,v):
    """
    From Balser 2011, Quireza 2006 APJ 653
    """

    return (7103.3 * v**1.1 / ratio / 1.08)**0.87
