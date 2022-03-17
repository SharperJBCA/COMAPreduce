# Classes describing individual sky model components 
#
import numpy as np
from comancpipeline.Simulations import FrequencyModels
from comancpipeline.Tools import WCS as cWCS
from comancpipeline.Tools import UnitConv
from astropy.io import fits
from astropy.wcs import WCS
import healpy as hp

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
        
        tod = self.get_map_values(gl, gb)
        
        tod *= self.frequency_model(frequency,**self.frequency_model_kwargs)

        return tod

    def get_map_values(self,gl,gb):
        """
        """
        pixels = cWCS.ang2pixWCS(self.wcs, gl, gb, self.skymap.shape)
        tod = self.skymap.flatten()[pixels]
        tod[(pixels == -1) | (tod == self.mapbadvalues) | ~np.isfinite(tod)] = 0

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

        
class HealpixSkyComponent(BasicSkyComponent):
    
    def read_skymap(self,mapfile):
        """
        Reads in the sky map, assumes it is a fits file
        --- replace this function for other sky component classes

        Converts all maps into units of K_RJ
        """

        m = hp.read_map(self.mapfile)
        if self.mapunit == 'MJyPixel':
            conv = UnitConv.Units('MJysr',self.mapfrequency)
            pixsize = 4*np.pi/m.size
            conv *= pixsize
        else:
            conv = UnitConv.Units(self.mapunit,self.mapfrequency)
        m *= conv
        self.nside = hp.npix2nside(m.size)
        return m, None

    def get_map_values(self,gl,gb):
        """
        """
        pixels = hp.ang2pix(self.nside, (90-gb)*np.pi/180., gl*np.pi/180.)
        tod = self.skymap[pixels]
        tod[(tod < self.mapbadvalues) | (tod == hp.UNSEEN)] = 0

        return tod
