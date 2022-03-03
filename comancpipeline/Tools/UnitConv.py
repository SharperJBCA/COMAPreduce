import numpy as np
from astropy import units as u
# constants
c = 299792458.
k = 1.3806488e-23
h = 6.62606957e-34
T_cmb = 2.725
Jy = 1e26

def toJy(_nu,beam):
    '''
    toJy(nu,beam)

    nu: frequency in GHz
    beam: beam in steradians.
    
    '''

    nu =_nu* 1e9
    return 2.*k*nu**2/c**2 * beam * Jy

def planckcorr(nu_in):

    nu = nu_in * 1e9

    x = h*nu/k/T_cmb

    return x**2*np.exp(x)/(np.exp(x) - 1.)**2

def Units(unit,nu,pixbeam=1):

    if isinstance(nu,type(None)):
        return 1

    conversions = {'K':1.,
                   'mK_RJ':1e-3,
                   'mK':1e-3,
                   'mKCMB':planckcorr(nu)*1e-3,
                   'KCMB':planckcorr(nu),
                   'Wm2sr':1,
                   'MJysr':1e6/toJy(nu,1.)}

    if unit in conversions:
        return conversions[unit]
    else:
        return 1

u.tau353 = u.def_unit('tau353',u.dimensionless_unscaled,format={'generic':r'$\tau_{353}$','latex':r'$\tau_{353}$'})
dustunits = {'Tau353':u.tau353,
             'IRAS100':u.MJy/u.sr,
             'Radiance':u.MJy/u.sr,
             'FDS98':u.mK,
             'RadianceTd1':u.MJy/u.sr*u.K,
             'RadianceTd2':u.MJy/u.sr*u.K**2,
             'Planck353': u.mK,
             'Planck857': u.MJy/u.sr}

dustscale = {'Tau353':1,
             'IRAS100':1e6*Units('MJysr',3000.),
             'Radiance':1,
             'FDS98':1,
             'RadianceTd1':1,
             'RadianceTd2':1,
             'Planck353': 1,
             'Planck857': 1}


duststrings = {'Tau353':r'K/$\tau_{353}$',
               'IRAS100':r'$\mu$K/$\mathrm{I}_{100\mu \mathrm{m}}$',
               'Radiance':r'K/$\mathcal{R}$',
               'FDS98':r'$\mu$K/mK',
               'RadianceTd1':r'K/$\mathcal{R}$K',
               'RadianceTd2':r'K/$\mathcal{R}$K$^{2}$',
               'Planck353':r'$\mu$K/$\mathrm{I}_{353}$' ,
               'Planck857': r'$\mu$K/$\mathrm{I}_{857}$'}

dusttitles = {'Tau353':r'$\tau_{353}$',
               'IRAS100':r'$\mathrm{I}_{100\mu \mathrm{m}}$',
               'Radiance':r'$\mathcal{R}$',
               'FDS98':r'FDS8',
               'RadianceTd1':r'$\mathcal{R} T_{D}$',
               'RadianceTd2':r'$\mathcal{R} T_{D}^{2}$',
               'Planck353':r'$\mathrm{I}_{353}$' ,
               'Planck857': r'$\mathrm{I}_{857}$'}


maptitles= {'Tau353':r'$\tau_{353}$',
            'IRAS100':r'$\mathrm{I}_{100\mu \mathrm{m}}$',
            'Radiance':r'$\mathcal{R}$',
            'FDS98':r'FDS8',
            'RadianceTd1':r'$\mathcal{R} T_{D}$',
               'RadianceTd2':r'$\mathcal{R} T_{D}^{2}$',
            'Planck353':r'$\mathrm{I}_{353}$' ,
            'Planck857': r'$\mathrm{I}_{857}$',
            'haslam':r'408MHz',
            'cbass':r'C-BASS',
            'cbass-subsrc':r'C-BASS',
            'wmapK':r'WMAP 22.8\,GHz',
            'Planck30':r'Planck 28.4\,GHz'}
            


syncscale = {'haslam':1e6,
             'cbass':1e3,
             'cbass-subsrc':1e3}
syncstrings = {'haslam':r'$\mu$K/K',
             'cbass':r'mK/K',
             'cbass-subsrc':r'mK/K'}
synctitles = {'haslam':r'408MHz',
             'cbass':r'C-BASS',
             'cbass-subsrc':r'C-BASS'}


halscale = {'HalphaWHAMSS':1e6,
            'HalphaDDD':1e6,
            'HalphaFDS':1e6}
halstrings = {'HalphaWHAMSS':r'$\mu$K/R',
              'HalphaDDD':r'$\mu$K/R',
              'HalphaFDS':r'$\mu$K/R'}

offscale = 1e6
offstrings = '$\mu$K'

duststrunits = {'Tau353':r'K/$\tau_{353}$',
                'IRAS100':r'$\mu$K/$\mathrm{MJy} \mathrm{ sr}^{-1}$',
                'Radiance':r'K/($\mathrm{W} \mathrm{m}^{-2} \mathrm{ sr}^{-1}$)',
                'FDS98':r'K/K',
                'RadianceTd1':r'K/$\mathrm{MJy} \mathrm{sr}^{-1}$K',
                'RadianceTd2':r'K/$\mathrm{MJy} \mathrm{ sr}^{-1}$K$^{2}$',
                'Planck353':r'K/K' ,
                'Planck857': r'$\mu$K/$\mathrm{MJy} \mathrm{ sr}^{-1}$'}


def dust_table_units(dmap, unit_now, unit_input):

    factors = {'Tau353':1,
                'IRAS100':1e6 * Units(unit_input,3000.) ,
                'Radiance':1 *Units(unit_input,5000.),
                'FDS98':1,
                'Planck353':1  ,
                'Planck857': 1e6 * Units(unit_input,857.) }

    unit = duststrunits[dmap]
    print(dmap, unit, factors[dmap],unit_now,unit_input)
    return unit, factors[dmap]

def dust_labels(labels):
    """
    Convert dust map data labels to a nice publication format
    """
    if isinstance(labels,list) | isinstance(labels,np.ndarray):
        return [dusttitles[d] for d in labels]
    else:
        return dusttitles[labels]

def BlackBody(nu,Td):
    """
    nu - in GHz
    Td - in K

    returns in Jy/sr
    """
    s = 1e9
    A = 2 * h * s**3/c**2
    B = np.exp(h*s*nu/k/Td)
    return A * nu**3 / (B-1) * Jy






def CheckUnit(unit,nu,pixbeam):
    if isinstance(nu,type(None)):
        return unit

    conversions = {'K':1.,
                   'mK_RJ':1e-3,
                   'mK':1e-3,
                   'mKCMB':planckcorr(nu)*1e-3,
                   'KCMB':planckcorr(nu),
                   'Wm2sr':1,
                   'MJysr':1e6/toJy(nu,1.)}

    if unit in conversions:
        return 'K'
    else:
        return unit

def GetUnits(dset):
    """
    Returns map units from processed map hdf5 dataset
    """

    return dset.attrs['unit'], dset.attrs['input_unit']

def SetUnits(nu, value, units1, units2):
    """
    Arguments:
    nu - frequency in GHz
    value - the fitted amplitude to be converted units of Unit1/Unit2
    units1 - List containing [CurrentUnit,TargetUnit] (numerator)
    units2 - List containing [CurrentUnit,TargetUnit] (denomenator)
    """

    f1 = Units(units1[0],nu)/Units(units1[1],nu)
    f2 = Units(units2[0],nu)/Units(units2[1],nu)
    return value * f1/f2
