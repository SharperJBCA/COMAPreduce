from mapext.core.srcClass import astroSrc
import astropy.units as u
from astropy.coordinates import SkyCoord

calibrators = {

    'CasA': astroSrc(
        name='Cassiopeia A',
        coord=SkyCoord(111.734751, -02.129568,
                       frame='galactic', unit='degree'),
        notes=['SHRTNAME:Cas A',
               'FITSMODE:mapext/data/models/CassiopeiaA_model.fits']
    ),

    'TauA': astroSrc(
        name='Taurus A',
        coord=SkyCoord(184.5586, -05.7848,
                       frame='galactic', unit='degree'),
        notes=['SHRTNAME: Tau A',
               'OTHRNAME: Crab Nebula; Messier 1',
               'OTHRSHRT:CRAB NEB; M1',
               'FITSMODE:mapext/data/models/TaurusA.fits']
    )

}
