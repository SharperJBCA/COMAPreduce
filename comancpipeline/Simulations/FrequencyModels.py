# Frequency models used for the sky model components
#

import numpy as np


def constant_scalar(frequency,constant=1,**kwargs):
    """
    Simplest function, return a constant
    """
    return constant

def powerlaw(frequency, A0=1, nu0=1, index=-2, **kwargs):
    """
    Scale by a power law relative to nu0
    """

    return A0*(frequency/nu0)**index

def blackbody(frequency, Td=20, beta=1.59,A0=1,nu0=353.,**kwargs):
    """
    Relative blackbody - frequency in GHz
    """
    hk = 4.7994e-2 # planck/boltzmann * 1e9
    y = hk/Td 
    top = np.exp(y*nu0)
    bot = np.exp(y*frequency)
    
    return A0*(frequency/nu0)**(beta+1) * top/bot 

