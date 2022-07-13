#!/usr/bin/env python3
#===============================================================================
# SciptName     : mapClass.py
# ScriptDir     : /src/core/mapClass.py
# Author(s)     : T.J.Rennie
# Description   : astroMap class for holding astronomical maps in processing
#===============================================================================

# P-I: IMPORTS
from astropy.io import fits
from astropy_healpix import HEALPix
import astropy.units as u
from astropy.wcs import WCS
from astropy.cosmology import Planck15
import healpy
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from skimage.measure import block_reduce
from copy import deepcopy
import reproject
import os
from tqdm import tqdm
from copy import deepcopy

from mapext.core.usefulFunctions import set_lims_inplace

# I: PROCESS CLASS


class astroMap():

    def __init__(self,
                 survey=None, name=None, reference=None,
                 filename=None,
                 frequency=None, wavelength=None, beamwidth=None,
                 resolution=None, projection=None, unit=None, error={},
                 note=[],
                 test=False,
                 load_map=True):
        if test:  # TEST VARIABLES FOR PIPELINE CHECKS
            # NAMES
            self.SURV = 'TEST'
            self.NAME = 'TEST'
            self.REFS = 'Rennie 2022'
            # FILE REFERENCES
            self.FILE = 'PLACEHOLDER'
            # OBSERVATION
            self.FREQ = 30.5e9*u.Hz
            self.WLEN = 3e8*(u.m*u.Hz)/self.FREQ
            self.BEAM = [4.5*u.arcmin, 4.5*u.arcmin]
            self.BMEA = (np.pi*self.BEAM[0]*self.BEAM[1]).to(u.sr)
            self.BAND = 1e9*u.Hz
            # MAP INFO
            self.RESO = 4.5*u.arcmin
            self.PROJ = 'CART'  # CART or HPX
            self.ERRS = {'CALIBRATION': 1*u.pc}
            # AOB
            self.NOTE = note
            # MAP
            self.MAP = np.ones((101, 201))*(u.MJy/u.sr)
            self.WCS = WCS(naxis=2)
            self.WCS.wcs.cdelt = [1/60, 1/60]
            self.WCS.wcs.crpix = [100, 50]
            self.WCS.wcs.crval = [30, 0]
            self.WCS.wcs.ctype = ['GLON-CYP', 'GLAT-CYP']
        else:
            # NAMES
            self.SURV = survey
            self.NAME = name
            self.REFS = reference
            # FILE REFERENCES
            self.FILE = filename
            # OBSERVATION
            isdata = [frequency != None, wavelength != None]
            if isdata[0] == True:
                self.FREQ = frequency
                self.WLEN = 3e8*(u.m*u.Hz)/self.FREQ
            elif isdata[1] and isdata[0] == False:
                self.WLEN = wavelength
                self.FREQ = 3e8*(u.m*u.Hz)/self.WLEN
            elif isdata[0] == False and isdata[1] == False:
                self.WLEN = None
                self.FREQ = None
            self.BEAM = beamwidth
            self.BMEA = (np.pi*self.BEAM[0]*self.BEAM[1]).to(u.sr)
            # MAP INFO
            if resolution != None:
                self.RESO = resolution
            else:
                self.RESO = np.sqrt(self.BEAM[0]*self.BEAM[1])
            self.PROJ = projection
            self.ERRS = error
            # AOB
            self.NOTE = note
            if load_map:
                self.WCS, self.MAP = self.load_map()
                if unit == 'KCMB':
                    self.MAP *= u.K
                    self.UNIT = 'KCMB'
                elif isinstance(unit, str):
                    self.MAP *= u.pc
                    self.UNIT = unit
                else:
                    self.MAP *= unit.value
                    self.MAP *= unit.unit
                    self.UNIT = self.MAP.unit
            else:
                self.WCS, self.MAP, self.UNIT = None, None, None
        self.ID = self.SURV+'_'+self.NAME
        self.IDshort = ''.join(
            [c for c in self.SURV+self.NAME if c.isupper() or c.isnumeric()])

    def ERRF(self, data):
        e = deepcopy(self.ERRS)
        total = 0
        for _ in e.keys():
            if e[_].unit == u.pc:
                e[_] = (e[_].value/100)*data
            total = total+e[_]**2
        total = np.sqrt(total)
        return np.array(total)

    def load_map(self,
                 n_wcs=0,
                 n_map=0):
        if self.PROJ == 'WCS':
            f = fits.open(self.FILE)
            map = f[n_map].data[:]
            map[map == 0] = np.nan
            wcs = WCS(f[n_wcs].header)[:]
        elif self.PROJ == 'WCS3':
            f = fits.open(self.FILE)
            map = np.array(f[n_map].data)[0, :, :]
            map[map == 0] = np.nan
            map[np.isfinite(map) == False] = np.nan
            h = f[0].header
            wcs = WCS(naxis=2)
            wcs.wcs.cdelt = [h['CDELT1'], h['CDELT2']]
            wcs.wcs.crpix = [h['CRPIX1'], h['CRPIX2']]
            wcs.wcs.crval = [h['CRVAL1'], h['CRVAL2']]
            wcs.wcs.ctype = [h['CTYPE1'], h['CTYPE2']]
        elif self.PROJ[:3] == 'HPX':
            f = fits.open(self.FILE)
            wcs = {}
            wcs['NSIDE'] = f[1].header['NSIDE']
            wcs['ORDER'] = f[1].header['ORDERING'].lower()
            if self.SURV.lower() == 'parkes':
                # print('PARKES: FORCE ORDERING:RING')
                wcs['ORDER'] = 'ring'
            map = np.array(healpy.read_map(self.FILE))
            # map[map <= -1.5e30] = np.nan
            map[map == 0] = np.nan
        return wcs, map

    def convert_unit(self,
                     units=u.Jy/u.sr,
                     update=True,
                     ignore=True):
        if units == self.UNIT:
            map_final = self.MAP[:]
            cont = False
        else:
            cont = True
            # first convert to Jy.sr
            if self.UNIT in [u.W/(u.m**2 * u.sr)]:
                map_int = self.MAP.to(
                    u.Jy/u.sr, equivalencies=u.spectral_density(self.FREQ))[:]
            elif self.UNIT in [u.K]:
                map_int = self.MAP.to(
                    u.Jy/u.sr, equivalencies=u.brightness_temperature(self.FREQ))[:]
            elif self.UNIT in [u.Jy/u.sr]:
                map_int = self.MAP[:]
            elif self.UNIT in [u.Jy/u.beam]:
                map_int = self.MAP.to(
                    u.Jy/u.sr, equivalencies=u.beam_angular_area(self.BMEA))[:]
            elif self.UNIT == 'KCMB':
                print('KCMB')
                map_int = self.MAP.to(
                    u.Jy/u.sr, equivalencies=u.thermodynamic_temperature(self.FREQ, Planck15.Tcmb0))
            else:
                print('UNIT NOT RECOGNISED {}'.format(self.UNIT))
                map_final = self.MAP[:]
                cont = False

            if cont:
                # convert to final units
                # first convert to Jy.sr
                if units in [u.W/(u.m**2 * u.sr)]:
                    map_final = map_int.to(
                        units, equivalencies=u.spectral_density(self.FREQ))[:]
                elif units in [u.K]:
                    map_final = map_int.to(
                        units, equivalencies=u.brightness_temperature(self.FREQ))[:]
                elif units in [u.Jy/u.sr]:
                    map_final = map_int[:]
                elif units in [u.Jy/u.beam]:
                    map_final = map_int.to(
                        units, equivalencies=u.beam_angular_area(self.BMEA))[:]
                elif units in [u.Jy/(u.pixel**2)]:
                    map_final = map_int.value * u.Jy * \
                        abs(np.radians(
                            self.WCS.wcs.cdelt[0])*np.radians(self.WCS.wcs.cdelt[1]))/(u.pixel**2)
                else:
                    print('UNIT NOT RECOGNISED {}, ASSUME 1'.format(self.UNIT))
                    map_final = map_int[:]

            # remove intermediate map and output
                del map_int

        if update:
            self.MAP = map_final
            if cont == False:
                units = self.UNIT
            self.NOTE.append(
                'UNIT CONVERTED FROM {} TO {}'.format(self.UNIT, units))
            self.UNIT = units
            return
        else:
            return map_final, self.WCS

    def rtn_coords(self):
        if type(self.WCS) is dict:
            hp = HEALPix(nside=self.WCS['NSIDE'],
                         order=self.WCS['ORDER'],
                         frame='galactic')
            coords = hp.healpix_to_skycoord(np.arange(self.map.shape[0]))
        else:
            x, y = np.meshgrid(np.arange(self.MAP.shape[1]),
                               np.arange(self.MAP.shape[0]),
                               sparse=True)
            coords = self.WCS.pixel_to_world(x, y)

        return coords

    def downsample(self,
                   target_pix=1*u.arcmin,
                   minScale=1,
                   update_map=True):
        current_pix = abs(self.WCS.wcs.cdelt[0])*u.degree.to(u.arcmin)
        sf = int(
            (target_pix/(self.WCS.wcs.cdelt[0]*u.degree)).decompose().value)
        if sf > minScale:
            print('Downscaling by a factor of {}'.format(sf))
            new_map = block_reduce(self.MAP.value,
                                   block_size=(sf, sf),
                                   func=np.nanmean,
                                   cval=np.nan)*self.UNIT
            new_wcs = WCS(naxis=2)
            new_wcs.wcs.crpix = self.WCS.wcs.crpix/sf
            new_wcs.wcs.cdelt = self.WCS.wcs.cdelt*sf
            new_wcs.wcs.crval = self.WCS.wcs.crval
            new_wcs.wcs.ctype = self.WCS.wcs.ctype
        else:
            if self.genParams['verbose']:
                print('NO RESCALING REQUIRED')

        if update_map:
            self.MAP = new_map
            self.NOTE.append('MAP DOWNSAMPLED FROM {} TO {}'.format(
                current_pix, target_pix))
            self.WCS = new_wcs
            return
        else:
            return new_map, new_wcs

    def smooth_to(self,
                  target_res=5*u.arcmin,
                  update_map=True):
        map_in = deepcopy(self.MAP[:])
        if self.RESO < target_res:
            map_in[np.isfinite(map_in) == False] = 0.
            std = np.sqrt((target_res**2-self.RESO**2)/(8*np.log(2)))
            sd = abs((std.to(u.degree).value)/self.WCS.wcs.cdelt[0])
            kernel = Gaussian2DKernel(x_stddev=sd)
            map_final = convolve(map_in, kernel)
            map_final[map_in == 0.] = np.nan
        elif self.RESO == target_res:
            return self.MAP, target_res
        else:
            map_final = np.zeros(map_in.shape)
            map_final[:] = np.nan
            raise ValueError('target resolution higher than map resolution')
        if update_map:
            self.MAP = map_final
            self.NOTE.append(
                'MAP SMOOTHED FROM {} TO {}'.format(self.RESO, target_res))
            self.RESO = target_res
            return
        else:
            return map_final, target_res

    def quickplot(self, map_kwargs={}):
        plt.figure()
        ax = plt.subplot(projection=self.WCS)
        ax.imshow(self.MAP.value, **map_kwargs, vmin=-0.05, vmax=10)
        ax.coords['glon'].set_major_formatter('d.dd')
        ax.coords['glat'].set_major_formatter('d.dd')
        plt.show()
        plt.close()

    def reproject(self,
                  cdelt=None, ctype=None, crpix=None, crval=None, wcs=None, shape_out=None,
                  healpix=None, nside=2048, order='ring',
                  update=True):
        # define initial projection
        if type(self.WCS) is WCS:
            initial_type = 'wcs'
        else:
            initial_type = 'hpx'
        # define final projection
        if healpix == None:
            final_type = 'wcs'
            # create new wcs to go to
            if wcs == None:
                new_wcs = WCS(naxis=2)
                if cdelt != None:
                    new_wcs.wcs.cdelt = cdelt
                else:
                    new_wcs.wcs.cdelt = self.WCS.wcs.cdelt

                if ctype != None:
                    new_wcs.wcs.ctype = ctype
                else:
                    new_wcs.wcs.ctype = self.WCS.wcs.ctype

                if crpix != None:
                    new_wcs.wcs.crpix = crpix
                else:
                    new_wcs.wcs.crpix = self.WCS.wcs.crpix

                if crval != None:
                    new_wcs.wcs.crval = crval
                else:
                    new_wcs.wcs.crval = self.WCS.wcs.crval
            else:
                new_wcs = wcs

            if shape_out == None:
                shape_out = [int(new_wcs.wcs.crpix[1]*2-1),
                             int(new_wcs.wcs.crpix[0]*2-1)]
        else:
            final_type = 'hpx'
            new_wcs = {}
            new_wcs['NSIDE'] = healpix
            new_wcs['ORDERING'] = order

        # case-by-case sampling
        if (initial_type == 'wcs') & (final_type == 'wcs'):
            new_map, footprint = reproject.reproject_interp((self.MAP.value,
                                                            self.WCS),
                                                            new_wcs,
                                                            shape_out=shape_out)

        elif (initial_type == 'wcs') & (final_type == 'hpx'):
            new_map, footprint = reproject.reproject_to_healpix((self.MAP.value,
                                                                self.WCS),
                                                                coord_system_out='galactic',
                                                                nside=new_wcs['NSIDE'],
                                                                nested=new_wcs['ORDER'] == 'nested')

        elif (initial_type == 'hpx') & (final_type == 'wcs'):
            new_y, new_x = np.meshgrid(np.arange(shape_out[0]),
                                       np.arange(shape_out[1]),
                                       sparse=True)
            new_coords = new_wcs.pixel_to_world(new_x, new_y)
            #HEALPIX projection
            hp = HEALPix(nside=self.WCS['NSIDE'],
                         order='RING',  # self.WCS['ORDER'],
                         frame='galactic')
            new_coords_hpx = hp.lonlat_to_healpix(new_coords.l,
                                                  new_coords.b)
            new_map = np.zeros(shape_out)
            new_map[:] = np.nan
            m = self.MAP.value
            m[m == 0] = np.nan
            m[np.isfinite(m) == False] = np.nan
            if self.MAP[0] != False:
                new_map = self.MAP.value[new_coords_hpx]
            else:
                hpx1 = fits.open(self.FILE)[1].data
                idx = hpx1['PIXEL']
                data = hpx1['SIGNAL']
                new_map = np.zeros(shape_out)
                i = np.searchsorted(idx, new_coords_hpx).astype(int)
                new_map = data[i]
            new_map = new_map.T
            footprint = np.isfinite(new_map)

            # new_map,footprint=reproject.reproject_from_healpix(self.FILE,
            #                                                    new_wcs,
            #                                                    shape_out=shape_out)
        else:
            print('TRANSFORM NOT CURRENTLY SUPPORTED')
            return

        new_map[footprint == 0] = np.nan
        if self.SURV == 'EFFELSBERG_GPS':
            new_map[new_map <= -30] = np.nan
        new_map *= self.MAP.unit

        if update:
            self.MAP = new_map
            self.WCS = new_wcs
            self.NOTE.append('MAP REPROJECTED TO CURRENT WCS')
            return
        else:
            return new_map, new_wcs

    def return_dictionary(self):
        out_dict = {'general':    {'SURV':    self.SURV,
                                   'NAME':    self.NAME,
                                   'REFS':    self.REFS,
                                   'FILE':    self.FILE},
                    'spectral':   {'FREQ':    self.FREQ.to(u.Hz).value,
                                   'WLEN':    self.WLEN.to(u.m).value},
                    'beam':       {'BEAM':    [self.BEAM[0].to(u.arcmin).value,
                                               self.BEAM[1].to(u.arcmin).value],
                                   'BMEA':    self.BMEA.to(u.sr).value},
                    'mapspace':   {'RESO':    self.RESO.to(u.arcmin).value,
                                   'PROJ':    self.PROJ,
                                   'ERRS':    self.ERRS},
                    'note':        np.array(self.NOTE, dtype='<S200')}
        return out_dict

    def save_out(self, ow=True, suffix='', filename=None):
        hdu = fits.PrimaryHDU(self.MAP.value, header=self.WCS.to_header())
        hdul = fits.HDUList(hdu)
        # Add keywords
        kw = self.return_dictionary()
        for _ in ['general', 'spectral', 'beam', 'mapspace']:
            for __ in kw[_]:
                hdul[0].header[__] = str(kw[_][__])
        for _l, l in enumerate(self.NOTE):
            hdul[0].header['COMMENT'] = self.NOTE[_l]
        # Save out
        if filename is not None:
            outfilename = filename
        else:
            outfilename = '{}-{}{}.fits'.format(self.ID, self.RESO, suffix)
        try:
            hdul.writeto(outfilename, overwrite=ow)
        except:
            print('File already exists - not overwritten')
        del hdul
        return
