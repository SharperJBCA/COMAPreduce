# Models for describing different calibrator sources
import numpy as np
from comancpipeline.Tools import Coordinates 

kb = 1.38064852e-23
c  = 3e-1 # Gm/second

# At 5.2 AU, ref Weiland et al 2011
nujwmap = np.array([22.8, 33., 40.9, 61., 93.8])
Tjwmap  = np.array([135.2, 146.6, 154.7, 165., 172.3])
jupfit  = np.poly1d(np.polyfit(np.log(nujwmap), Tjwmap, 2))
jupAng0 = 2.481e-8 # sr

def JupiterFlux(nu, mjd, lon=0, lat=0, source='jupiter'):
    """
    """

    r0, d0, dist = Coordinates.getPlanetPosition(source, lon, lat, mjd)

    jupAng = jupAng0*(5.2/dist)**2
    
    return 2. * kb * (nu/c)**2 * jupfit(np.log(nu)) * jupAng * 1e26, dist
