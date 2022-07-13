
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from mapext.core.processClass import Process
from mapext.core.srcClass import astroSrc

_default_linestyles = {b'ApPhoto':  {'marker':'^'},
                       b'2DGaussFit':{'marker':'v'},
                       b'literature':{'marker':'*'}}
_default_colors = ['black','tab:purple','tab:green','tab:blue']

class plot_spectrum(Process):

    def run(self,src):
        hilight_survey = [b'COMAP',b'COMAP Legacy',b'Cruciani 2016']

        flux_types = list(set(list(src.flux['method'])))
        plt.figure()
        for t in flux_types:
            msk = src.flux['method']==t
            print(src.flux)
            for _hs,hs in enumerate(hilight_survey):

                Dmsk = src.flux['survey']==hs
                msk_temp = np.all([Dmsk,msk],axis=0)
                plt.errorbar(src.flux['nu'][msk_temp]/1e9,
                             src.flux['Sv'][msk_temp],
                             yerr=src.flux['Sv_e'][msk_temp],
                             ls='none',
                             c=_default_colors[_hs+1],
                             **_default_linestyles[t])
                msk[msk_temp] = False


            plt.errorbar(src.flux['nu'][msk]/1e9,
                         src.flux['Sv'][msk],
                         yerr=src.flux['Sv_e'][msk],
                         label=t.decode(),
                         ls='none',
                         c=_default_colors[0],
                         **_default_linestyles[t])
        plt.loglog()
        plt.legend()
        plt.xlabel(r'$\nu$ [GHz]')
        plt.ylabel(r'$S_\nu$ [Jy]')
        plt.show()
