# Classes describing individual sky model components 
#
import numpy as np
from comancpipeline.Simulations import FrequencyModels
from comancpipeline.Tools import WCS as cWCS
from comancpipeline.Tools import UnitConv
from astropy.io import fits
from astropy.wcs import WCS

class BasicSkyComponent:

    def __init__(self, mapfile='', frequency_model='',
                 mapbadvalues=-1e32,
                 mapunit='',mapfrequency=1, **kwargs):
        """
        """

        self.mapfile = mapfile
        self.mapfrequency = mapfrequency
        self.mapunit = mapunit
        self.mapbadvalues = mapbadvalues
        self.frequency_model = FrequencyModels.__dict__[frequency_model]
        self.frequency_model_kwargs = kwargs

        self.skymap, self.wcs = self.read_skymap(mapfile)


    def __call__(self, gl, gb, frequency):
        """
        
        """

        pixels = cWCS.ang2pixWCS(self.wcs, gl, gb, self.skymap.shape)
        tod = self.skymap.flatten()[pixels]
        tod[(pixels == -1) | (tod < self.mapbadvalues)] = 0

        tod *= self.frequency_model(frequency,**self.frequency_model_kwargs)
        return tod

    def read_skymap(self,mapfile):
        """
        Reads in the sky map, assumes it is a fits file
        --- replace this function for other sky component classes

        Converts all maps into units of K_RJ
        """

        hdu = fits.open(mapfile)
        if self.mapunit == 'MJyPixel':
            conv = UnitConv.Units('MJysr',self.mapfrequency)
            pixsize = (hdu[0].header['CDELT1']*np.pi/180.)**2
            conv *= pixsize
        else:
            conv = UnitConv.Units(self.mapunit,self.mapfrequency)

        m = hdu[0].data[...]*conv
        wcs = WCS(hdu[0].header)
        hdu.close()

        return m, wcs

        
