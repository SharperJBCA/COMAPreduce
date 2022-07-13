from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from reproject import reproject_interp
from astropy.wcs import wcs
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle

from mapext.core.processClass import Process
from mapext.core.mapClass import astroMap
from mapext.core.srcClass import astroSrc
from mapext.io.outFile import outFileHandler
from mapext.core.usefulFunctions import set_lims_inplace
from matplotlib.gridspec import GridSpec
from cmcrameri import cm
import os

spath = os.path.abspath(os.path.dirname(__file__))
tab20_clrs = np.loadtxt(
    '{}/../core/tab20.txt'.format(spath), delimiter=' ')/255
tab10_clrs = np.loadtxt(
    '{}/../core/tab10.txt'.format(spath), delimiter=' ')/255


class cutout(Process):

    def run(self, map, src_lst, dist=2*u.degree, plot=False):
        # load map in Jy/pix units
        MAP, WCS = map.convert_unit(units=u.Jy/u.sr, update=False)

        shape_out = [int(dist.to(u.arcmin).value//1*2 + 1),
                     int(dist.to(u.arcmin).value//1*2 + 1)]
        crpix_out = [int(dist.to(u.arcmin).value//1),
                     int(dist.to(u.arcmin).value//1)]

        for s in src_lst:
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.cdelt = [-1/60, 1/60]
            wcs_out.wcs.crval = [s.COORD.l.degree, s.COORD.b.degree]
            wcs_out.wcs.crpix = crpix_out
            wcs_out.wcs.ctype = ['GLON-CYP', 'GLAT-CYP']
            map_out, footprint = reproject_interp(
                (MAP.value, WCS), wcs_out, shape_out=shape_out)
            map_out *= MAP.unit

            self.RUNDICT['OUTFILE'].save_map(s, map_out, wcs_out, map.ID)

        if plot:
            plt.figure()
            ax = plt.subplot(projection=wcs_out)
            ax.imshow(map_out.value, **set_lims_inplace(map_out.value))
            plt.show()


class cutoutgrid(Process):

    def run(self, src_lst, map_lst, globallims=False, circles=[12, 16, 20], srcScale=False):
        circles = np.array(circles)
        for src in src_lst:
            print('mapping {}'.format(src.NAME))
            maps = []
            freq = []
            wcss = []
            for m in map_lst:
                try:
                    map, unit, wcs = self.RUNDICT['OUTFILE'].rtrv_map(
                        src, m.ID)
                except:
                    continue
                map[map == 0] = np.nan

                try:
                    self.validateMap(map)
                except:
                    continue

                maps.append(map)
                freq.append(m.FREQ.value/1e9)
                wcss.append(wcs)

            try:
                freq, maps = zip(*sorted(zip(freq, maps)))
                maps = np.array(maps)
                print('done')
                avg = np.nanmedian(maps, axis=(-1, -2))
                maps -= avg[:, np.newaxis, np.newaxis]
                print('done')

                if globallims == True:
                    lims = set_lims_inplace(
                        maps.flatten(), max_step=1e5, lower=0.18)

                plt.figure(figsize=(2*2, 2*2))
                gs = GridSpec(2, 2,
                              # left=0.05,right=0.95,
                              # bottom=0.05,top=0.95,
                              hspace=0, wspace=0)

                options = {'cmap': cm.roma_r}

                for idx in range(len(freq)):
                    ax = plt.subplot(gs[idx//2, idx % 2], projection=wcss[idx])
                    if globallims == False:
                        ax.imshow(maps[idx], **options,
                                  **set_lims_inplace(maps[idx], max_step=1e5, lower=0.05, upper=0.99))
                    else:
                        ax.imshow(maps[idx], **options, **lims)
                    if freq[idx] < 100:
                        t = ax.text(0.0, 0.0, '{:1.2f} GHz'.format(
                            freq[idx]), ha='left', va='bottom', transform=ax.transAxes, c='white')
                    else:
                        t = ax.text(0.0, 0.0, '{:1.2f} THz'.format(
                            freq[idx]/1e3), ha='left', va='bottom', transform=ax.transAxes, c='white')

                    if srcScale:
                        radii = circles*src.RAD.to(u.arcmin).value
                    else:
                        radii = circles

                    for _ in range(2, int(radii[-1]//2*2), 2):
                        c = Circle((wcss[idx].wcs.crval[0],
                                    wcss[idx].wcs.crval[1]),
                                   _/60,
                                   edgecolor='black',
                                   facecolor='none',
                                   alpha=0.3,
                                   transform=ax.get_transform('galactic'))
                        ax.add_patch(c)

                    for _ in radii:
                        c = Circle((wcss[idx].wcs.crval[0],
                                    wcss[idx].wcs.crval[1]),
                                   _/60,
                                   edgecolor='yellow',
                                   facecolor='none',
                                   transform=ax.get_transform('galactic'))
                        ax.add_patch(c)

                    lon, lat = ax.coords[0], ax.coords[1]
                    lon.set_major_formatter('d.dd')
                    lat.set_major_formatter('d.dd')
                    # lon.set_ticks_visible(False)
                    # lon.set_ticklabel_visible(False)
                    # lat.set_ticks_visible(False)
                    # lat.set_ticklabel_visible(False)
                    # lon.set_axislabel('')
                    # lat.set_axislabel('')

                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(self.RUNDICT['PLOTDIR']+src.NAME
                            + '_CUTOUT_{}.pdf'.format(circles[0]))
            except:
                print('Fail')

    def validateMap(self, map):
        if np.sum(np.isfinite(map)) == 0:
            raise Error('No data in map')
        if map[0, 0] == np.nanmean(map):
            raise Error('Map is flat')


class map_with_srcs(Process):

    def run(self, map, srcs):
        # load map in Jy/pix units
        MAP, WCS = map.convert_unit(units=u.Jy/(u.pixel**2), update=False)

        plt.figure()
        ax = plt.subplot(projection=WCS)

        plt.imshow(MAP.value, cmap='Greys',
                   **set_lims_inplace(MAP.value, lower=0.05))
        # if type(srcs) is dict:
        for __, _ in enumerate(srcs.keys()):
            slst = srcs[_]
            for _s, s in enumerate(slst):
                if _s == 0:
                    plt.scatter(s.COORD.l.degree, s.COORD.b.degree,
                                c=[tab10_clrs[__]], s=10, transform=ax.get_transform('world'), label=_)
                else:
                    plt.scatter(s.COORD.l.degree, s.COORD.b.degree,
                                c=[tab10_clrs[__]], s=10, transform=ax.get_transform('world'))
        plt.legend()
        plt.show()


class interest_map(Process):

    def run(self, map, sigma=3.0):

        MAP, WCS = map.convert_unit(units=u.Jy/(u.pixel**2), update=False)

        H_elems = hessian_matrix(MAP.value+1e3, sigma=sigma, order='rc')
        srcs, ridges = hessian_matrix_eigvals(H_elems)

        fig, ax = plt.subplots(
            1, 2, sharex=True, sharey=True, subplot_kw={'projection': WCS})
        ax[0].imshow(-ridges, **set_lims_inplace(-ridges,
                     sym=True), cmap='bwr')
        ax[1].imshow(-srcs, **set_lims_inplace(-srcs, sym=True), cmap='bwr')

        ax[0].contour(MAP.value, levels=60, colors='black', lw=0.5,
                      **set_lims_inplace(MAP.value))
        ax[1].contour(MAP.value, levels=60, colors='black', lw=0.5,
                      **set_lims_inplace(MAP.value))

        ax[0].set_title('Structure')
        ax[1].set_title('Sources')

        plt.show()
