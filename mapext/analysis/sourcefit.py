
import astropy.wcs as wcs
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from tqdm import tqdm
import h5py
from astropy.modeling import models, fitting
import emcee
import corner
from scipy.optimize import minimize
import multiprocessing as mp
mp.set_start_method('fork')
import time

from mapext.core.processClass import Process
from mapext.core.mapClass import astroMap
from mapext.core.srcClass import astroSrc

class single_source(Process):

    def run(self,map,src_lst,burnin=2500,runtime=1000,fit_radius=15*u.arcmin,Nwalker=50):
        # load map in Jy/pix units
        MAP, WCS = map.convert_unit(units=u.Jy/(u.pixel**2),update=False)
        # ensure src_lst is a list of sources and retrieve coordinates of each one
        if type(src_lst) is astroSrc:
            src_lst = [src_lst]
        map_skyCoord = map.rtn_coords()

        map_noise = 0.01

        for _s,s in enumerate(src_lst):
            x_pc,y_pc = wcs.utils.skycoord_to_pixel(s.COORD,map.WCS)
            x_pc,y_pc = int(x_pc),int(y_pc)
            D=int(abs(fit_radius.to(u.degree).value/WCS.wcs.cdelt[0])+3)
            map_cutout = MAP[y_pc-D:y_pc+D+1,x_pc-D:x_pc+D+1].value
            coords_cutout = map.rtn_coords()[y_pc-D:y_pc+D+1,x_pc-D:x_pc+D+1]

            wcs_cutout = wcs.WCS(naxis=2)
            wcs_cutout.wcs.cdelt = WCS.wcs.cdelt
            wcs_cutout.wcs.crval = WCS.wcs.crval
            wcs_cutout.wcs.ctype = WCS.wcs.ctype
            wcs_cutout.wcs.crpix = [WCS.wcs.crpix[0]-(x_pc-D),WCS.wcs.crpix[1]-(y_pc-D)]

            x_cutout = coords_cutout.l.degree
            y_cutout = coords_cutout.b.degree
            dx_cutout = x_cutout-s.COORD.l.degree
            dy_cutout = y_cutout-s.COORD.b.degree
            dr_cutout = np.sqrt(dx_cutout**2 + dy_cutout**2)*u.degree
            fitting_area = dr_cutout<fit_radius

            p0 = [np.nanmedian(map_cutout),   # intercept
                  0.0,                  # slope_x
                  0.0,                  # slope_y
                  1,                    # Amplitude
                  0.0,                  # x_mean
                  0.0,                  # y_mean
                  5/(60*np.sqrt(8*np.log(2))),              # x_stddev
                  5/(60*np.sqrt(8*np.log(2))),                # ratio
                  0]                 # epsilon

            nll = lambda *args: -lnlike(*args)
            soln = minimize(nll, p0, args=(dx_cutout, dy_cutout, map_cutout, map_noise, fitting_area),
                            bounds = ((0,np.inf),(None,None),(None,None),
                                      (1e-2,np.inf),
                                      (-2/60,2/60),(-2/60,2/60),
                                      (1.6/60,5./60),(1.6/60,5./60),
                                      (-np.pi/2,np.pi/2)))

            # fig,axs = plt.subplots(nrows=2,ncols=2)
            # axs[0,0].imshow(map_cutout)
            # axs[0,1].imshow(gaussian_model1(soln.x[:-1],dx_cutout,dy_cutout)[0])
            # axs[1,1].imshow(map_cutout-gaussian_model1_s(soln.x[:-1],dx_cutout,dy_cutout)[0])
            # plt.show()

            Ndim = len(p0)
            P0 = [soln.x + 1e-11*np.random.randn(Ndim) for i in range(Nwalker)]
            pool = mp.Pool()
            sampler = emcee.EnsembleSampler(Nwalker, Ndim, lnprob,
                                            args=(dx_cutout,
                                                  dy_cutout,
                                                  map_cutout,
                                                  map_noise,
                                                  fitting_area),
                                            pool=pool)
            state = sampler.run_mcmc(P0,burnin+runtime,skip_initial_state_check=True,progress=True)
            pool.close()

            pars = [r'$c$',r'$m_x$',r'$m_y$',r'$A$',r'$x_c$',r'$y_c$',r'$a$',r'$b$',r'$\theta$']

            chains = sampler.chain[:]

            # sort a>b
            mask = chains[:,:,6]<chains[:,:,7] #filter where a<b
            lower = chains[:,:,6][mask]
            chains[:,:,6][mask] = chains[:,:,7][mask]
            chains[:,:,7][mask] = lower
            chains[:,:,8][mask] -= np.pi/2

            # collapse angles into mode±pi/2 range
            chains[:,:,8] -= chains[:,:,8]//np.pi * np.pi
            theta_hist, edge = np.histogram(chains[:,:,8], bins=20, range=(0, np.pi))
            cen = (edge[1:]+edge[:-1])/2
            max_index = np.argmax(theta_hist)
            cval = cen[max_index]
            chains[:,:,8][chains[:,:,8]<cval-np.pi/2] += np.pi
            chains[:,:,8][chains[:,:,8]>cval+np.pi/2] -= np.pi

            def create_filter(chain,n,thresh=1.5):
                final_amps = chain[:,-1,n]
                q1 = np.percentile(final_amps, 25, interpolation='midpoint')
                q3 = np.percentile(final_amps, 75, interpolation='midpoint')
                iqr = q3-q1
                lower, upper = q1-(thresh*iqr), q3+(thresh*iqr)
                walkermask = np.all([final_amps>=lower, final_amps<=upper],axis=0)
                return walkermask

            walkermask = np.all([create_filter(chains,4),
                                 create_filter(chains,4)],
                                axis=0)
            print('discard ',np.sum(walkermask==False),' walkers of ',Nwalker)

            chains_collM = np.nanmean(chains[walkermask],axis=0)
            chains_coll = np.nanmedian(chains[walkermask],axis=0)
            soln2 = np.nanmean(chains_coll[burnin:],axis=0)
            errs2 = np.nanstd(chains_coll[burnin:],axis=0)

            # OUTPUT MODEL INFO
            src_model_entry = [(map.SURV,map.NAME,
                                (map.FREQ/u.Hz).value,
                                soln2[3:],errs2[3:])]
            s.add_src_model(src_model_entry)

            Sv = 2*np.pi*soln2[3]*soln2[6]*soln2[7]*(60**2)*u.Jy
            Sv_e = Sv*np.sqrt((errs2[3]/soln2[3])**2 + ((errs2[6]/soln2[6])**2) + (errs2[7]/soln2[7])**2)
            print(_s,' : Sv = ',Sv.value,'±',Sv_e.value)

            # OUTPUT FLUX INFO
            flux_info = [('2DGaussFit',map.SURV,map.NAME,
                          (map.FREQ/u.Hz).value,Sv.value,Sv_e.value)]
            s.add_flux(flux_info)

            # # BACKUP TXT FILE
            # with open('{}_{}_sfit.txt'.format(s.NAME,map.NAME), 'a') as the_file:
            #     the_file.write('Sv = {} ± {}'.format(Sv.value,Sv_e.value))

            # GRAPHS AND output

            samples_flat = sampler.chain[walkermask, burnin:, :].reshape((-1, Ndim))
            fig = corner.corner(samples_flat, labels=pars, truths=soln2, quantiles=[0.25,0.5,0.75])
            plt.savefig('{}_{}_cornerplot.pdf'.format(s.NAME,map.NAME))
            plt.close()

            fig, axs = plt.subplots(ncols=3,nrows=4)
            for n in range(Ndim):
                ax = axs[n//3,n%3]
                ax.plot(chains[:,:,n].T, c='black', alpha = 0.2)
                ax.plot(chains_collM[:,n].T, c='blue')
                ax.plot(chains_coll[:,n].T, c='red')
                ax.set_title(pars[n])
                ax.set_ylim(np.nanmin(chains_coll[:,n]),np.nanmax(chains_coll[:,n]))
            axs[-1,0].scatter(chains_collM[:,4],chains_collM[:,5],c=np.arange(burnin+runtime),s=1)
            axs[-1,0].set_title('center')
            axs[-1,1].scatter(chains_collM[:,6],chains_collM[:,7],c=np.arange(burnin+runtime),s=1)
            plt.savefig('{}_{}_chains.pdf'.format(s.NAME,map.NAME))
            plt.close()

            # fig,axs = plt.subplots(nrows=2,ncols=2)
            # axs[0,0].imshow(map_cutout)
            # axs[0,1].imshow(gaussian_model1(soln.x[:-1],dx_cutout,dy_cutout)[0])
            # axs[1,0].imshow(map_cutout-gaussian_model1_s(soln.x[:-1],dx_cutout,dy_cutout)[0])
            # axs[1,1].imshow(gaussian_model1_s(soln.x[:-1],dx_cutout,dy_cutout)[1])

            fig,axs = plt.subplots(nrows=2,ncols=2,subplot_kw={'projection':wcs_cutout})
            im1 = axs[0,0].imshow(map_cutout,origin='lower')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$b$')
            plt.colorbar(im1,ax=axs[0,0])
            im2 = axs[0,1].imshow(gaussian_model1(soln2,dx_cutout,dy_cutout),origin='lower')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$b$')
            plt.colorbar(im2,ax=axs[0,1])
            im3 = axs[1,0].imshow(map_cutout-gaussian_model1_s(soln2,dx_cutout,dy_cutout),origin='lower')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$b$')
            plt.colorbar(im3,ax=axs[1,0])
            im4 = axs[1,1].imshow(gaussian_model1_s(soln2,dx_cutout,dy_cutout),origin='lower')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$b$')
            plt.colorbar(im4,ax=axs[1,1])

            plt.savefig('{}_{}_model.pdf'.format(s.NAME,map.NAME))
            plt.close()

def lnprior(P):
    c,mx,my,A,xc,yc,a,b,t = P
    if np.all([A>1e-4,
               abs(xc)<=2.5/60.,
               abs(yc)<=2.5/60.,
               a>1.5/60., a<7./60.,
               b>1.5/60., b<7./60.],axis=0):
        return 0.
    return -np.inf

def lnlike(P,x,y,map,map_noise,fitting_area):
    model = gaussian_model1(P,x,y)
    denom = np.power(map_noise,2)
    lp = -0.5*np.nansum(np.power(map[fitting_area]-model[fitting_area],2)/denom + np.log(denom) + np.log(2*np.pi))
    # corr = (mask.shape[0]*mask.shape[1])/np.nansum(mask)
    # lp = lp*corr
    return lp

def lnprob(P,x,y,map,map_noise,fitting_area):
    lp = lnprior(P)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnlike(P,x,y,map,map_noise,fitting_area)
    return lp+ll

def gaussian_model1(P,x,y):
    i,sx,sy,A,x0,y0,xs,ys,t = P
    BGD = models.Planar2D(intercept=i,
                          slope_x=sx,
                          slope_y=sy)
    SRC = models.Gaussian2D(amplitude=A,
                            x_mean=x0,
                            y_mean=y0,
                            x_stddev=xs,
                            y_stddev=ys,
                            theta=t)
    # MASK = SRC(x,y) >= A*0.1
    MOD = BGD(x,y) + SRC(x,y)
    return MOD#, MASK

def gaussian_model1_s(P,x,y):
    i,sx,sy,A,x0,y0,xs,ys,t = P
    SRC = models.Gaussian2D(amplitude=A,
                            x_mean=x0,
                            y_mean=y0,
                            x_stddev=xs,
                            y_stddev=ys,
                            theta=t)
    # MASK = SRC(x,y) >= A*0.1
    MOD = SRC(x,y)
    return MOD#, MASK
