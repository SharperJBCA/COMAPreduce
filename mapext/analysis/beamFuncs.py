import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from tqdm import tqdm
from astropy.convolution import convolve
from skimage.restoration import richardson_lucy, unsupervised_wiener
from scipy import fftpack
from copy import deepcopy as copy
from scipy.integrate import simps
from scipy.signal import find_peaks, detrend

import matplotlib.pyplot as plt

from mapext.core.processClass import Process
from mapext.core.mapClass import astroMap
from mapext.data.Calibrators import calibrators
import astropy.units as u


def gaussiandb(r, s):
    val = np.exp(-0.5 * (r)**2 / s**2)
    db = 10*np.log10(val)
    return db


class createBeam(Process):

    def run(self, m, beamname='BEAM'):

        Riter = 10
        psiz = 40
        mres = 0.25 * m.RESO.to(u.arcmin).value

        cals = self.return_calibrators(m)

        for cal in cals.items():
            new_calcoord = cal[1].COORD

            for n in range(1):
                calcoord = new_calcoord
                print(n, ' : ', calcoord)
                calcoord = np.array(m.WCS.world_to_pixel(calcoord)).astype(int)

                calmap = m.MAP.value[calcoord[1]-Riter: calcoord[1]+Riter,
                                     calcoord[0]-Riter: calcoord[0]+Riter]

                ixs = np.array(np.meshgrid(
                    np.arange(calmap.shape[0]), np.arange(calmap.shape[1])))

                wcen = np.sum(calmap[np.newaxis, :, :]*ixs,
                              axis=(1, 2)) / np.sum(calmap)

                new_calcoord = m.WCS.pixel_to_world(
                    wcen[0]+calcoord[0]-Riter, wcen[1]+calcoord[1]-Riter)

            print(calcoord)

            model_fname = 'mapext/data/models/{}_model.fits'.format(
                cal[1].NAME.replace(' ', ''))

            model = astroMap(survey='Bonn', name=cal[0],
                             filename=model_fname,
                             beamwidth=[0*u.arcmin, 0*u.arcmin],
                             projection='WCS',
                             unit=1*u.Jy)
            model.reproject(wcs=m.WCS, shape_out=m.MAP.shape)
            model.MAP[np.isnan(model.MAP)] = 0

            calmap = m.MAP.value[calcoord[1]-psiz: calcoord[1]+psiz+1,
                                 calcoord[0]-psiz: calcoord[0]+psiz+1]
            ixs = np.array(np.meshgrid(
                np.arange(calmap.shape[0]), np.arange(calmap.shape[1])))
            # model.quickplot()

            psf = self.deconvolve(calmap,
                                  model.MAP.value[calcoord[1]-psiz: calcoord[1]+psiz+1,
                                                  calcoord[0]-psiz: calcoord[0]+psiz+1])
            psf /= np.nanmax(psf)

            np.savetxt('{}_BEAMMAP.txt'.format(m.ID, beamname), psf)

            # fit for center by taking STD in area around peak
            xx, yy = np.meshgrid(np.arange(psf.shape[0]),
                                 np.arange(psf.shape[1]))

            weight = psf**2
            xc, yc = np.nansum(xx*weight) / \
                np.nansum(weight), np.nansum(yy*weight)/np.nansum(weight)

            print(xc, yc)
            xx, yy = np.meshgrid(np.arange(psf.shape[0])-xc,
                                 np.arange(psf.shape[1])-yc)
            rr = np.sqrt(xx*xx + yy*yy)
            tt = np.arctan2(yy, xx)

            mask = np.all([tt > -0.5, tt < 2.3,
                           rr > 2.0, rr < 100], axis=0) == False

            binvals, binedges, bincount = binned_statistic(
                rr.flatten(), psf.flatten(), bins=np.linspace(0, 100, 101)*mres, statistic='mean')
            binq1, _, __ = binned_statistic(
                rr.flatten(), psf.flatten(), bins=np.linspace(0, 100, 101)*mres, statistic=lambda x: np.percentile(x, 25))
            binq3, _, __ = binned_statistic(
                rr.flatten(), psf.flatten(), bins=np.linspace(0, 100, 101)*mres, statistic=lambda x: np.percentile(x, 75))
            binstd, _, __ = binned_statistic(
                rr.flatten(), psf.flatten(), bins=np.linspace(0, 100, 101)*mres, statistic=lambda x: np.nanstd(x))

            binctrs = 0.5*(binedges[:-1]+binedges[1:])

            def normfunc(r, p):
                db = gaussiandb(r, m.RESO.to(u.arcmin).value/2.355)
                return db + p

            print('!!!', m.RESO.to(u.arcmin).value*(2.5/2.355))

            fit = np.all([np.isfinite(10*np.log10(binvals)),
                          10*np.log10(binvals) > 10*np.log10(np.nanmax(binvals))-10], axis=0)
            p, c = curve_fit(normfunc, binctrs[fit], 10*np.log10(binvals[fit]))

            beamout = np.array(
                [np.append([0], 0.5*(binedges[:-1]+binedges[1:])),
                 np.append([0], 10*np.log10(binvals)-p)])

            #interpolate nan values in beam model:
            beamout[1, np.isnan(beamout[1])] = np.interp(beamout[0, np.isnan(
                beamout[1])], beamout[0, np.isfinite(beamout[1])], beamout[1, np.isfinite(beamout[1])])
            beamout2 = np.array([rr.flatten(), psf.flatten()])

            np.savetxt('{}_{}.txt'.format(m.ID, beamname), beamout)
            np.savetxt('{}_RAW_{}.txt'.format(m.ID, beamname), beamout2)

            plt.figure()
            ax = plt.subplot()

            ax.scatter(rr.flatten(), 10*np.log10(psf.flatten())
                       - p, c='k', alpha=0.1, s=1)

            plt.fill_between(0.5*(binedges[:-1]+binedges[1:]), 10*np.log10(
                binq1)-p, 10*np.log10(binq3)-p, facecolor='tab:blue', alpha=0.2)
            ax.fill_between(beamout[0, 1:], beamout[1, 1:]-binstd,
                            beamout[1, 1:]+binstd, facecolor='tab:blue', alpha=0.2)
            ax.plot(beamout[0], beamout[1], c='tab:blue')

            X = np.linspace(0, 50, 301)
            ax.plot(X, 10*np.log10(np.exp(-0.5*X*X/((m.RESO.to(u.arcmin).value/2.355)**2))),
                    color='tab:green')

            ax.set_xlabel(r'$r$')
            ax.set_ylabel(r'$I_r$ [dB]')

            ax.set_xlim(0, 50)
            ax.set_ylim(-45, 3)

            ax.axvspan(35, 100, alpha=0.1, color='tab:red')

            # MAIN BEAM RATIO
            beam_real = 10**(beamout[1]/10)
            gaussmod_real = np.exp(-0.5*beamout[0]*beamout[0]
                                   / ((m.RESO.to(u.arcmin).value/2.355)**2))
            mask = beamout[0] < 35
            mainbeam = simps((gaussmod_real[mask])*beamout[0][mask],
                             beamout[0][mask])
            sidelobes = simps((beam_real[mask]-gaussmod_real[mask])*beamout[0][mask],
                              beamout[0][mask])

            err = simps((binstd[mask[1:]])*beamout[0]
                        [mask][1:], beamout[0][mask][1:])/sidelobes
            err = err*mainbeam/sidelobes*0.01

            print('MAINBEAM: ', mainbeam)
            print('SIDELOBES: ', sidelobes)
            print('S/m: ', sidelobes/mainbeam, ' ± ', err)

            # PEAKS:
            ydat = 10*np.log10(np.abs(beam_real-gaussmod_real))[mask]
            dataloc = np.where(np.isfinite(ydat))[0]
            gradient = (ydat[dataloc[0]]-ydat[dataloc[-1]]) / \
                (dataloc[0]-dataloc[-1])
            c = ydat[dataloc[0]]-dataloc[0]*gradient
            ydat = ydat - gradient*np.arange(ydat.shape[0]) - c

            peaks, _ = find_peaks(ydat)
            delt = 0
            for __, _ in enumerate(peaks):
                if beamout[0][_] > m.RESO.to(u.arcmin).value*1.2:
                    ax.scatter(beamout[0][_],
                               beamout[1][_] + 2,
                               marker='v',
                               c='tab:orange')
                    ax.text(beamout[0][_],
                            beamout[1][_] + 3,
                            'SL{}\n{:2.3f} dB'.format(
                                __+1-delt, beamout[1][_]),
                            color='tab:orange', va='bottom', ha='center')
                    ax.text(30, 2,
                            'MAINBEAM-FULLBEAM RATIO:\n{:4.3f} ± {:4.3f}'.format(mainbeam
                                                                                 / (mainbeam+sidelobes),
                                                                                 err),
                            ha='center', va='top')
                else:
                    delt += 1

            # plt.show()
            plt.savefig('{}_{}.pdf'.format(m.ID, beamname))

    def return_calibrators(self, m):
        local_cals = {}
        for cal in calibrators.items():
            pix = np.array(m.WCS.world_to_pixel(cal[1].COORD))
            if np.any([pix < 0, pix > m.MAP.shape[::-1]]):
                continue
            testval = m.MAP.value[int(pix[1]), int(pix[0])]
            if np.isfinite(testval) is False:
                continue
            local_cals[cal[0]] = cal[1]
        return local_cals

    def deconvolve(self, m, model, method='lr'):
        if method == 'fft':
            m_fft = fftpack.fftshift(fftpack.fftn(m))
            model_fft = fftpack.fftshift(fftpack.fftn(model))
            psf = fftpack.fftshift(fftpack.ifftn(
                fftpack.ifftshift(m_fft/model_fft)))
            psf = np.abs(psf)

        elif method in ['lr', 'rl']:
            psf = richardson_lucy(m, model, clip=False)

        elif method == 'wh':
            print(type(m), type(model))
            psf, _ = unsupervised_wiener(
                np.array(m), np.array(model), clip=False)

        return psf


class correctBeam(Process):

    def run(self, m, beamname='BEAM', kernelsize=30*u.arcmin):
        beam = np.loadtxt('{}_{}.txt'.format(m.ID, beamname))
        pbar = tqdm(total=6, colour='green')

        kernelsize = 20

        # CREATE KERNELS
        xx, yy = np.meshgrid(np.linspace(-kernelsize, kernelsize, 2*kernelsize + 1),
                             np.linspace(-kernelsize, kernelsize, 2*kernelsize + 1))
        rr = np.sqrt(xx*xx + yy*yy)
        # BEAM KERNEL
        interpBeam = np.interp(rr, beam[0], beam[1])
        interpBeam = 10**(interpBeam/10)
        interpBeam = interpBeam/np.nanmax(interpBeam)
        interpBeam[rr > kernelsize] = 0
        pbar.update(1)
        #GAUSSIAN KERNEL
        gaussianMod = np.exp(-0.5*rr*rr/((m.RESO.to(u.arcmin).value/2.355)**2))
        pbar.update(1)

        # fb = np.loadtxt('/Volumes/TJRennie/MAPEXT22/EFF2.7-BEAM_FROM_FIG.txt')
        # plt.figure()
        # plt.scatter(rr, interpBeam, c='r')
        # plt.scatter(rr, gaussianMod, c='b')
        # plt.scatter(fb[0], 10**(fb[1]/10), c='k')
        # plt.show()

        # IMPORT MAP
        intmap = copy(m.MAP.value)
        intmap -= np.nanmin(intmap)
        intmap = np.array(intmap).astype(float)
        intmap[np.isfinite(intmap) == False] = 0
        pbar.update(1)

        # DO CONVOLUTION

        intmap1 = richardson_lucy(intmap, interpBeam, clip=False)
        pbar.update(1)
        intmap1 = convolve(intmap1, gaussianMod)
        pbar.update(1)

        intmap1[np.isfinite(m.MAP.value) == False] = np.nan
        intmap1[m.MAP.value == 0] = np.nan
        pbar.update(1)

        m.MAP = intmap1*m.MAP.unit
        m.NOTE.append(
            'BEAM CORRECTED USING {}_{}.txt'.format(m.ID, beamname))
