
import astropy.wcs as wcs
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

from mapext.core.processClass import Process
from mapext.core.mapClass import astroMap
from mapext.core.srcClass import astroSrc
from mapext.core.regClass import astroReg
from mapext.core.usefulFunctions import set_lims_inplace
from matplotlib import path
from scipy.ndimage.measurements import center_of_mass
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, 0, 0), (1, 0, 0), (0, 0, 1)]  # K -> R -> B
cmap = LinearSegmentedColormap.from_list('KRB3', colors, N=3)


class ap_source(Process):

    def run(self, map, src_lst, apertures=10*u.arcmin):

        # load map in Jy/pix units
        MAP, WCS = map.convert_unit(units=u.Jy/(u.pixel**2), update=False)

        # ensure src_lst is a list of sources and retrieve coordinates of each one
        if type(src_lst) is astroSrc:
            src_lst = [src_lst]
        src_skyCoord = [src.COORD for src in src_lst]
        map_skyCoord = map.rtn_coords()

        coords_r = np.array(
            [map_skyCoord.separation(sc).degree for sc in src_skyCoord])
        mask = np.zeros(coords_r.shape).astype(int)

        if apertures != 'src':
            apertures_deg = apertures.to(u.degree).value
            mask[coords_r <= apertures_deg] = 1
            mask[np.all([coords_r >= apertures_deg*(4/3),
                        coords_r <= apertures_deg*(5/3)], axis=0)] = 2

        else:
            rads = np.array([src.RAD.to(u.degree).value for src in src_lst])
            mask[coords_r <= rads[:, np.newaxis, np.newaxis]] = 1
            mask[np.all([coords_r >= rads[:, np.newaxis, np.newaxis]*(4/3),
                        coords_r <= rads[:, np.newaxis, np.newaxis]*(5/3)], axis=0)] = 2

        ap_count = np.sum(mask == 1, axis=(1, 2))*u.pixel**2
        an_count = np.sum(mask == 2, axis=(1, 2))*u.pixel**2

        ap = (mask == 1).astype(float)*MAP[np.newaxis, :, :]
        ap[ap == 0] = np.nan

        an = (mask == 2).astype(float)*MAP[np.newaxis, :, :]
        an[an == 0] = np.nan

        ap_sum = np.nansum(ap, axis=(1, 2))*u.pixel**2
        an_mean = np.nanmean(an, axis=(1, 2))
        an_med = np.nanmedian(an, axis=(1, 2))
        an_std = np.nanstd(an, axis=(1, 2))

        # print(ap_sum.shape,ap_sum[0])
        # print(an_mean.shape,an_mean[0])
        # print(an_med.shape,an_med[0])
        # print(an_std.shape,an_std[0])

        Sv = ap_sum - an_med*ap_count
        Sv_e = (an_std*np.sqrt(an_count + ap_count))*u.pixel
        # print('!',Sv_e**2)
        # print('!',map.ERRF(Sv)**2)
        # print('!',Sv_e**2 + (map.ERRF(Sv)*u.Jy)**2)
        Sv_e = np.sqrt(Sv_e**2 + map.ERRF(Sv)**2*(u.Jy**2))

        for _ in range(Sv.shape[0]):
            print(_, ' : Sv = ', Sv[_].value, '±', Sv_e[_].value)
            # OUTPUT FLUX INFO
            if Sv[_].value != 0.0:
                flux_info = [('ApPhoto', map.SURV, map.NAME,
                              (map.FREQ/u.Hz).value, Sv[_].value, Sv_e[_].value)]
                src_lst[_].add_flux(flux_info)

            # with open('{}_{}_apho.txt'.format(src_lst[_].NAME,map.NAME), 'a') as the_file:
            #     the_file.write('Sv = {} ± {}'.format(Sv[_].value,Sv_e[_].value))


class ap_region(Process):

    def run(self, map, reg_lst, apertures=10*u.arcmin):

        # load map in Jy/pix units
        MAP, WCS = map.convert_unit(units=u.Jy/(u.pixel**2), update=False)
        map_skyCoord = map.rtn_coords()
        c = np.array([map_skyCoord.l.degree, map_skyCoord.b.degree])

        # ensure src_lst is a list of sources and retrieve coordinates of each one
        if type(reg_lst) is astroReg:
            reg_lst = [reg_lst]

        mask = np.zeros([len(reg_lst), MAP.shape[0], MAP.shape[1]])

        for _reg, reg in enumerate(reg_lst):
            # retrieve fluxes
            choose = np.nan
            modelnu = reg.reg_bounds['nu']
            # choose a model as:
            # 1. where nu_model = nu_map
            # 2. where difference between nu_model and nu_map is least in log-space
            if map.FREQ.value in modelnu:
                choose = np.where(map.FREQ.value == modelnu)[0][0]
                # print('1',choose)
            else:
                diff = (np.log10(map.FREQ.value)-np.log10(modelnu))**2
                choose = np.where(np.nanmin(diff) == diff)[0][0]
                # print('2',choose)

            bkg = SkyCoord(
                *reg.reg_bounds[choose]['bkg'][:2], frame='galactic', unit='degree')
            coords_r = np.array(map_skyCoord.separation(bkg).degree)
            mask[_reg][coords_r < reg.reg_bounds[choose]['bkg'][2]] = 2

            model = reg.reg_bounds[choose]['bkg2']
            _val = np.where(np.isfinite(model[:, 0]) == False)[0][0]
            model = model[:_val]
            region_query = path.Path(np.array(model))
            isin = region_query.contains_points(
                c.reshape([2, c.shape[2]*c.shape[1]]).T)
            mask[_reg][isin.reshape([c.shape[1], c.shape[2]])] = 2

            model = reg.reg_bounds[choose]['bounds']
            _val = np.where(np.isfinite(model[:, 0]) == False)[0][0]
            model = model[:_val]
            region_query = path.Path(np.array(model))
            isin = region_query.contains_points(
                c.reshape([2, c.shape[2]*c.shape[1]]).T)
            mask[_reg][isin.reshape([c.shape[1], c.shape[2]])] = 1

            # plotting
            c_ap = center_of_mass(mask[_reg] == 1)
            c_an = center_of_mass(mask[_reg] == 2)
            delt = 1/WCS.wcs.cdelt
            ax = plt.subplot(projection=WCS)
            im = ax.imshow(MAP.value, cmap='Greys',
                           ** set_lims_inplace(MAP.value, lower=0.02))
            ax.imshow(mask[_reg], cmap=cmap, alpha=0.3,
                      interpolation='nearest')
            ax.plot(model[:, 0], model[:, 1], c='red', lw=1,
                    transform=ax.get_transform('galactic'))
            ax.text(c_ap[1], c_ap[0], 'SRC', va='center', ha='center')
            ax.text(c_an[1], c_an[0], 'BKG', va='center', ha='center')
            plt.ylim(c_ap[0]-delt[0], c_ap[0]+delt[0])
            plt.xlim(c_ap[1]-delt[1], c_ap[1]+delt[1])
            plt.colorbar(im, label=MAP.unit)
            plt.savefig('./A-'+map.SURV+'-'+map.NAME+'.pdf')
            plt.close()

        ap_count = np.sum(mask == 1, axis=(1, 2))
        an_count = np.sum(mask == 2, axis=(1, 2))

        ap = (mask == 1).astype(float)*MAP[np.newaxis, :, :]
        ap[ap == 0] = np.nan

        an = (mask == 2).astype(float)*MAP[np.newaxis, :, :]
        an[an == 0] = np.nan

        ap_sum = np.nansum(ap, axis=(1, 2))*u.pixel**2
        an_mean = np.nanmean(an, axis=(1, 2))*u.pixel**2
        an_med = np.nanmedian(an, axis=(1, 2))*u.pixel**2
        an_std = np.nanstd(an, axis=(1, 2))*u.pixel**2

        Sv = ap_sum - an_med*ap_count
        Sv_e = (an_std*np.sqrt(an_count + ap_count))
        Sv_e = np.sqrt(Sv_e**2 + map.ERRF(Sv)**2)

        for _ in range(Sv.shape[0]):
            print(_, ' : Sv = ', Sv[_].value, '±', Sv_e[_].value)
            # # OUTPUT FLUX INFO
            if Sv[_].value != 0.0:
                flux_info = [('ApPhoto', map.SURV, map.NAME,
                              (map.FREQ/u.Hz).value, Sv[_].value, Sv_e[_].value)]
                reg_lst[_].add_flux(flux_info)

            # with open('{}_{}_apho.txt'.format(src_lst[_].NAME,map.NAME), 'a') as the_file:
            #     the_file.write('Sv = {} ± {}'.format(Sv[_].value,Sv_e[_].value))
