import numpy as np
import os
keys = {'wnm':'spdust2_wnm.dat',
        'wim':'spdust2_wim.dat',
        'rn':'spdust2_rn.dat',
        'pdr':'spdust2_pdr.dat',
        'mc':'spdust2_mc.dat',
        'dc':'spdust2_dc.dat',
        'cnm':'spdust2_cnm.dat'}

def get_spdust_model(name):
    """
    Read in SPDUST model and return it
    """
    path = os.path.dirname(os.path.realpath(__file__))
    
    model = np.loadtxt(f'{path}/{keys[name]}', skiprows=21).T
    
    return model
