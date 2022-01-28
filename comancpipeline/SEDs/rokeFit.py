#!/usr/bin/env python3
#===============================================================================
# SciptName     : rokeFit.py
# ScriptDir     : /src/output/rokeFit.py
# Author(s)     : T.J.Rennie
# Description   : Wrapper for the RCA MCMC SED fitter
#===============================================================================


# P-I: IMPORTS
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import numpy as np
import sys

# P-II: CUSTOM IMPORTS
from src.core.processClass import process
from src.core import physConst

# I: rokeFit
class rokeFit(process):
    __name__    = 'rokeFit'
    __parent__  = 'analysis.rokeFit'
    __author__  = 'TBD'
    __version__ = '0.0.1'
    __date__    = '15/06/2021'

    def run(self,*args, **kwargs):
        h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']))
        sources = list(h.keys())[:]
        h.close()
        status = ['1band','8band']
        print(sources)
        sources = []
        for s in tqdm(sources):
            h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']))
            n_surveys = len(list(h['{}/APflux'.format(s)].keys()))
            data = np.zeros((n_surveys),dtype=[('name','U50'),('nu','f8'),('S','f8'),('Se','f8')])
            for _,survey in enumerate(list(h['{}/APflux'.format(s)].keys())):
                data[_]['name'] = survey
                data[_]['nu'] = h['{}/APflux/{}'.format(s,survey)][0]
                data[_]['S'] = h['{}/APflux/{}'.format(s,survey)][1]
                data[_]['Se'] = h['{}/APflux/{}'.format(s,survey)][2]
            h.close()
            nu = data['nu']
            S = data['S']
            Se = data['Se']

            # fig = plt.figure()
            # ax = plt.subplot()
            # ax.scatter(nu/1e9,S)
            # ax.set_xlabel(r'$\nu$ [GHz]')
            # ax.set_xlabel(r'$S_\nu$ [Jy]')

            # TRIGGER
            sys.path.insert(0, '/Volumes/TJRennie/github/mcmc')
            import mcmc, emission, tools
            from config_mcmc import settings
            del sys.path[0]

            custom = {'source_name' : s,
                      'result_dir'  : self.genParams['outDir'],
                      'plot_dir'    : self.genParams['outDir'],
                      }

            settings['source_name'] = s
            settings['plotting']['resultdir'] = self.genParams['outDir']
            settings['plotting']['plotdir'] = self.genParams['outDir']
            # settings['components']['ame'] = 0

            # CORR = np.identity(len(nu))
            # Se_ = np.asarray(Se)[:,np.newaxis]
            # CORR = Se_*CORR*Se_.T

            # # try:
            # COMAP_corr = np.load('COMAP_correlation.npy')
            c_true = np.zeros(len(nu)).astype(bool)
            for _ in range(len(nu)):
                if data['name'][_][:5] == 'COMAP':
                    c_true[_] = True
            # a = COMAP_corr*np.mean(np.diag(CORR[c_true][:,c_true]))
            # CORR[4:12,4:12] = a

            mask1 = np.all([nu!=100e9,
                            nu!=217e9,
                            nu!=4997e9,
                            nu!=26.5,
                            nu!=27.5,
                            nu!=28.5,
                            nu!=30.5,
                            nu!=31.5,
                            nu!=32.5,
                            nu!=33.5,], axis=0)

            #np.all([nu<1e11,S>1e3],axis=0)==False,

            use = np.all([nu<4e12,np.isfinite(S),S>1e-1, mask1],axis=0)

            # try:
            mcmc_data, mcmc_model, mcmc_settings = mcmc.mcmc(nu[use]/1e9, S[use], Se[use],
                beam=np.radians(60./60.)**2, custom_settings=settings,
                excluded=[])
            # except:
            #     continue
            def model(x):
                return mcmc_model['sed_model'](x,
                                               np.radians(60./60.)**2 *np.pi/4,
                                               mcmc_model['sed_params'])
            fig = plt.figure()
            ax = plt.subplot()
            argn = 0 # first argument number
            style_rep = ['--', '-.', ':']
            for i in mcmc_model['components']['n_components']:
                component = mcmc_model['components']['function'][i]
                comp_nameparams = mcmc_model['components']['name_params'][i]
                comp_nparams = mcmc_model['components']['n_params'][i]
                args_to_pass = mcmc_model['sed_params'][argn:argn+comp_nparams]
                component_flux = component(10**np.arange(-1,4.01,0.01),np.radians(60./60.)**2 *np.pi/4,*args_to_pass) # call each function and add up the flux
                linestyle_to_use = np.mod(i,len(style_rep))
                ax.plot(10**np.arange(-1,4.01,0.01),component_flux,color="black",linestyle=style_rep[linestyle_to_use],alpha=0.5)
                argn += comp_nparams
            ax.plot(10**np.arange(-1,4.01,0.01),model(10**np.arange(-1,4.01,0.01)),c='red')
            ax.errorbar(nu[use]/1e9,S[use],yerr=Se[use],
                        marker='o',linestyle='none',ms=4,c='black')
            ax.errorbar(nu[use==False]/1e9,S[use==False],yerr=Se[use==False],
                        marker='o',fmt='o',mfc='white',c='black',ms=4)

            ax.set_xscale("log", nonpositive='clip')
            ax.set_yscale("log", nonpositive='clip')
            ax.set_title(s)
            ax.set_xlabel(r'$nu$ [GHz]')
            ax.set_ylabel(r'$S_\nu$ [Jy]')
            try:
                ax.set_ylim(10.**float((np.log10(np.nanmin(model(10**np.arange(-1,4.01,0.01))))//1.)),
                            10.**float((np.log10(np.nanmax(model(10**np.arange(-1,4.01,0.01))))//1. +1.)))
            except:
                ax.set_ylim(10**-2,10**3)
            ax.set_xlim((10**0,10**4.5))

            Inset_axes = ax.inset_axes([0.08,0.65,0.4,0.3])
            Inset_axes.plot(np.arange(26,34.01,0.02),model(np.arange(26,34.01,0.02)),c='red')
            # print(c_true)
            Inset_axes.errorbar(nu[c_true]/1e9,
                                S[c_true],
                                yerr=Se[c_true],
                                marker='o',linestyle='none',ms=4,c='black')
            Inset_axes.set_ylim((np.nanmin(S[c_true])-0.15,np.nanmax(S[c_true])+0.15))
            Inset_axes.set_xlim((25.5,34.5))

            outfilename = '{}/{}_SpectrumPlots_model_ONE_fit.png'.format(self.genParams['outDir'],s)
            plt.savefig(outfilename)
            plt.close()

            # chi2 = mcmc_model['sed_chi_squared']
            # self.ver_print('Chi2: {}'.format(chi2). ver_no=2)

            h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']),'a')

            # if '{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/redChi2'.format(s) in h:
            #     del h['{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/redChi2'.format(s)]
            # h['{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/redChi2'.format(s)] = chi2Red

            if '{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/params'.format(s) in h:
                del h['{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/params'.format(s)]
            h['{}/rokeFit/AME;FreeFree;ThDust(7)_model_ONE/params'.format(s)] = mcmc_model['sed_params']

            h.close()
            # except:
                # continue


        return True, None

# II: rokeFitCorr
class rokeFitCorr(process):
    __name__    = 'rokeFit'
    __parent__  = 'analysis.rokeFit'
    __author__  = 'TBD'
    __version__ = '0.0.1'
    __date__    = '15/06/2021'

    def run(self,resol = 30, *args, **kwargs):
        h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']))
        sources = list(h.keys())[:]
        h.close()
        status = ['1band','8band']
        print(sources)
        # sources = ['G030.750-00.020']
        for s in tqdm(sources[:]):
            h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']))
            n_surveys = len(list(h['{}/APflux'.format(s)].keys()))
            data = np.zeros((n_surveys),dtype=[('name','U50'),('nu','f8'),('S','f8'),('Se','f8')])
            for _,survey in enumerate(list(h['{}/APflux'.format(s)].keys())):
                data[_]['name'] = survey
                data[_]['nu'] = h['{}/APflux/{}'.format(s,survey)][0]
                data[_]['S'] = h['{}/APflux/{}'.format(s,survey)][1]
                data[_]['Se'] = h['{}/APflux/{}'.format(s,survey)][2]
            h.close()
            nu = data['nu']
            S = data['S']
            Se = data['Se']

            # TRIGGER
            sys.path.insert(0, '/Volumes/TJRennie/github/mcmcCorr')
            import mcmc, emission, tools
            from config_mcmc import settings
            del sys.path[0]

            custom = {'source_name' : s,
                      'result_dir'  : self.genParams['outDir'],
                      'plot_dir'    : self.genParams['outDir'],
                      }

            settings['source_name'] = s
            settings['plotting']['resultdir'] = self.genParams['outDir']
            settings['plotting']['plotdir'] = self.genParams['outDir']
            #
            settings['components']['ame'] = 0
            settings['components']['synchrotron'] = 1
            settings['components']['freefree'] = 0

            runname = 'FINAL_SNRfit2'

            CORR = np.identity(len(nu))
            Se_ = np.asarray(Se)[:,np.newaxis]
            CORR = Se_*CORR*Se_.T

            try:
                COMAP_corr = np.load('COMAP_correlation.npy')
                c_true = np.zeros(len(nu)).astype(bool)
                for _ in range(len(nu)):
                    if data['name'][_][:5] == 'COMAP':
                        c_true[_] = True
                a = COMAP_corr*np.mean(np.diag(CORR[c_true][:,c_true]))
                CORR[c_true][:,c_true] = a

                def chk(a,b,c):
                    return abs(a-b)>c

                mask1 = np.all([nu!=100e9,nu!=217e9,nu!=4997e9, nu!=10e9], axis=0)

                #np.all([nu<1e11,S>1e3],axis=0)==False,

                use = np.all([nu<400e9,np.isfinite(S),S>1e-3, mask1],axis=0)
                # print(nu[mask1])

                # print('!!!!!')
                # for n in use:
                #     if n:
                #         print(nu[n]/1e9,S[n],Se[n,n])

                # fig = plt.figure()
                # ax = plt.subplot()
                # # ax.scatter(nu[mask1]/1e9,S[mask1],c='red')
                # ax.scatter(nu[use]/1e9,S[use],c='red')
                # ax.scatter(nu[use==False]/1e9,S[use==False],c='blue')
                # ax.set_xlabel(r'$\nu$ [GHz]')
                # ax.set_xlabel(r'$S_\nu$ [Jy]')
                # plt.loglog()
                # plt.show()


                try:
                    mcmc_data, mcmc_model, mcmc_settings = mcmc.mcmc(nu[use]/1e9, S[use], CORR[use][:,use],
                        beam=np.radians(resol/60.)**2, custom_settings=settings,
                        excluded=[])
                except:
                    pass
                def model(x):
                    return mcmc_model['sed_model'](x,
                                                   np.radians(resol/60)**2,
                                                   mcmc_model['sed_params'])
                fig = plt.figure()
                ax = plt.subplot()
                argn = 0 # first argument number
                style_rep = ['--', '-.', ':']
                paramDict = {}
                for i in mcmc_model['components']['n_components']:
                    component = mcmc_model['components']['function'][i]
                    comp_nameparams = mcmc_model['components']['name_params'][i]
                    comp_nparams = mcmc_model['components']['n_params'][i]
                    args_to_pass = mcmc_model['sed_params'][argn:argn+comp_nparams]
                    component_flux = component(10**np.arange(-1,4.01,0.01),np.radians(resol/60)**2,*args_to_pass) # call each function and add up the flux
                    linestyle_to_use = np.mod(i,len(style_rep))
                    ax.plot(10**np.arange(-1,4.01,0.01),component_flux,color="black",linestyle=style_rep[linestyle_to_use],alpha=0.5)
                    argn += comp_nparams

                    fname = str(component).split(' ')[1]
                    paramDict[fname] = {}
                    paramDict[fname]['1'] = {}
                    for _ in range(len(args_to_pass)):
                        p = comp_nameparams[_]
                        v = args_to_pass[_]
                        paramDict[fname]['1'][p] = v
                ax.plot(10**np.arange(-1,4.01,0.01),model(10**np.arange(-1,4.01,0.01)),c='red')
                # ax.errorbar(nu[:]/1e9,S[:],yerr=Se[:],marker='x',linestyle='none',ms=4,c='black')

                ax.errorbar(nu[use]/1e9,S[use],yerr=Se[use],
                            marker='o',linestyle='none',ms=4,c='black')
                ax.errorbar(nu[use==False]/1e9,S[use==False],yerr=Se[use==False],
                            marker='o',fmt='o',mfc='white',c='black',ms=4)

                ax.set_xscale("log", nonpositive='clip')
                ax.set_yscale("log", nonpositive='clip')
                ax.set_title(s)
                ax.set_xlabel(r'$nu$ [GHz]')
                ax.set_ylabel(r'$S_\nu$ [Jy]')
                try:
                    ax.set_ylim(10.**float((np.log10(np.nanmin(model(10**np.arange(-1,4.01,0.01))))//1.)),
                                10.**float((np.log10(np.nanmax(model(10**np.arange(-1,4.01,0.01))))//1. +1.)))
                except:
                    ax.set_ylim(10**-2,10**3)
                ax.set_xlim((10**0,10**4.5))

                Inset_axes = ax.inset_axes([0.08,0.65,0.4,0.3])
                Inset_axes.plot(np.arange(26,34.01,0.02),model(np.arange(26,34.01,0.02)),c='red')
                Inset_axes.errorbar(nu[c_true]/1e9,S[c_true],yerr=Se[c_true],
                            marker='o',linestyle='none',ms=4,c='black')
                Inset_axes.set_ylim((np.nanmin(S[c_true])-0.15,np.nanmax(S[c_true])+0.15))
                Inset_axes.set_xlim((25.5,34.5))

                outfilename = '{}/{}_SpectrumPlots_{}_fit.png'.format(self.genParams['outDir'],s,runname)
                # plt.show()
                plt.savefig(outfilename)
                plt.close()

                # ndof = np.nansum(use) - len(mcmc_model['sed_params'])
                # print('ndof:  ',ndof)
                # chi2 = np.nansum((S[use]-model(nu[use]))**2/(np.diag(CORR)[use]), axis=0)
                # print('Chi2:         ',chi2)


                h = h5py.File('{}/EXTRACT.hd5'.format(self.genParams['outDir']),mode='a')

                # if '{}/rokeFit/AME;FreeFree;ThDust(7)_{}/redChi2'.format(s,runname) in h:
                #     del h['{}/rokeFit/AME;FreeFree;ThDust(7)_{}/redChi2'.format(s,runname)]
                # h['{}/rokeFit/AME;FreeFree;ThDust(7)_{}/redChi2'.format(s,runname)] = chi2

                # if '{}/rokeFit/AME;FreeFree;ThDust(7)_{}/params'.format(s,runname) in h:
                #     del h['{}/rokeFit/AME;FreeFree;ThDust(7)_{}/params'.format(s,runname)]
                # h['{}/rokeFit/AME;FreeFree;ThDust(7)_{}/params'.format(s,runname)] = mcmc_model['sed_params']

                def pushDict(h5file,obj):
                    if isinstance(obj, dict) and isinstance(h5file,h5py.Group):
                        for key, value in obj.items():
                            if isinstance(value, dict):
                                if key in h5file:
                                    del h5file[key]
                                if key not in h5file:
                                    h5file.create_group(key)
                                pushDict(h5file[key],value)
                            else:
                                if key in h5file:
                                    del h5file[key]
                                h5file[key] = value

                if '{}/Model/{}'.format(s,runname) in h:
                    del h['{}/Model/{}'.format(s,runname)]
                h.create_group('{}/Model/{}'.format(s,runname))
                h.create_group('{}/Model/{}/model_dict'.format(s,runname))
                pushDict(h['{}/Model/{}/model_dict'.format(s,runname)],paramDict)
                h['{}/Model/{}/fitted'.format(s,runname)] = use

            except:
                pass

            h.close()


        return True, None
