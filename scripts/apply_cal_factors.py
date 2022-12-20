# apply_cal_factors.py
# SH - 24/11/22
#
# Write to each level 3 file the correct calibration factors.
#

import numpy as np
from matplotlib import pyplot
import glob
import h5py
from tqdm import tqdm
from astropy.wcs import WCS
import os
import sys
from astropy.io import fits
from comancpipeline.Tools import Coordinates, binFuncs, CaliModels,UnitConv


badobs = {'<7000':[i for i in range(1,20)],
          '6042':[i for i in range(20)], # v. bad
          '7314':[5],
          '7425':[19],
          '22267':[10],
          '8415':[i for i in range(20)],
          '9358':[8],
          '9421':[i for i in range(20)],
          '9573':[i for i in range(20)],
          '9960':[i for i in range(20)],
          '10013':[i for i in range(20)],
          '10147':[i for i in range(20)],
          '10626':[i for i in range(20)],
          '10653':[i for i in range(20)],
          '10784':[i for i in range(20)],
          '11770':[i for i in range(20)],
          '11798':[i for i in range(20)],
          '11960':[i for i in range(20)],
          '11987':[i for i in range(20)],
          '12060':[i for i in range(20)],
          '12175':[i for i in range(20)],
          '12261':[i for i in range(20)],
          '12290':[i for i in range(20)], #Sun?
          '12431':[i for i in range(20)],
          '12460':[i for i in range(20)],
          '12488':[i for i in range(20)],
          '12515':[i for i in range(20)],
          '12546':[i for i in range(20)],
          '12606':[i for i in range(20)],
          '12733':[i for i in range(20)],
          '12764':[i for i in range(20)],
          '12866':[i for i in range(20)],
          '13650':[6], # spike
          '13795':[i for i in range(20)],
          '13851':[14], # spike
          '14023':[i for i in range(20)],
          '14081':[i for i in range(20)],
          '14139':[i for i in range(20)],
          '14167':[i for i in range(20)],
          '14195':[i for i in range(20)],
          '14225':[i for i in range(20)],
          '14254':[i for i in range(20)],
          '14282':[i for i in range(20)],
          '14479':[8],
          '14618':[i for i in range(20)],
          '14696':[14],
          '14726':[14],
          '14755':[14],
          '14813':[14],
          '14913-15086':[14],
          '15116':[18],
          '15453':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
          '15481':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
            '15502':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
            '15530':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
            '15559':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
            '15587':[4,10,12], # loss of sensitivity in 10? 4 and 12 swap
            '15615':[i for i in range(20)],
            '15642':[4,12], #  4 and 12 swap
            '15668':[4,12], #  4 and 12 swap
            '15697':[4,12], #  4 and 12 swap
            '15725':[4,12], #  4 and 12 swap
            '15754':[4,12], #  4 and 12 swap
            '15783':[4,12], #  4 and 12 swap
            '15811':[4,12], #  4 and 12 swap
            '15840':[4,12], #  4 and 12 swap
            '15868':[4,12], #  4 and 12 swap
            '15897':[4,12], #  4 and 12 swap
            '15925':[4,12], #  4 and 12 swap
            '15941':[4,12], #  4 and 12 swap
            '15982':[4,12], #  4 and 12 swap
            '16007':[4,12], #  4 and 12 swap
            '16030':[4,12], #  4 and 12 swap
            '16053':[4,12], #  4 and 12 swap
            '16077':[4,12], #  4 and 12 swap
            '16107':[i for i in range(20)], #  4 and 12 swap
            '16134':[i for i in range(20)], #  4 and 12 swap
            '16394':[5], 
            '16485':[i for i in range(20)], 
            '17214':[i for i in range(20)], 
            '17276':[i for i in range(20)], 
            '17330':[5], 
            '17432':[i for i in range(20)], 
            '17669':[18], # Weather event under feed 18
            '17722':[i for i in range(20)],  # Timing issue? 
            '17751':[i for i in range(20)],  # Timing issue? 
            '17780':[i for i in range(20)], 
            '18155':[7], 
            '18392':[i for i in range(20)], 
            '18421':[i for i in range(20)], 
            '18600':[i for i in range(20)], 
            '18630':[i for i in range(20)], 
            '18775':[i for i in range(20)], 
            '19094':[i for i in range(20)], 
            '19311':[i for i in range(20)], 
            '19368':[i for i in range(20)], 
            '19427':[i for i in range(20)], 
            '19457':[i for i in range(20)], 
            '19579':[i for i in range(20)], 
            '19669':[i for i in range(20)], # Weather event under feed 18
            '19699':[18], 
            '19760':[i for i in range(20)], 
            '19904':[17], # IQ cal messed up in 17?
            '20024':[i for i in range(20)], 
            '20232':[i for i in range(20)], 
            '20263':[i for i in range(20)], 
            '20293':[i for i in range(20)],  # v. bad
            '20336':[i for i in range(20)], 
            '20448':[i for i in range(20)], # receivers not working
                      '20475':[i for i in range(20)], # receivers not working
                      '20504':[i for i in range(20)], # Something strange with sources?
            '20784':[i for i in range(20)],  # v. bad
            '20915':[i for i in range(20)],  # v. bad
            '20947':[i for i in range(20)],  #  bad
            '20977':[i for i in range(20)],  #  bad
            '20998':[i for i in range(20)],  #  bad
            '21028':[i for i in range(20)],  #  bad
            '21058':[i for i in range(20)],  # v. bad
            '21113':[i for i in range(20)],  # v. bad
            '21168':[i for i in range(20)],  # v. bad
            '21198':[i for i in range(20)],  # v. bad
            '21870':[i for i in range(20)],  # bad
            '21943':[i for i in range(20)],  # bad -- some feeds IQcal bad?
            '21945':[i for i in range(20)],  # v. bad
            '21947':[i for i in range(20)],  # v. bad
            '21948':[i for i in range(20)],  # v. bad
            '21949':[i for i in range(20)],  # v. bad
            '21950':[i for i in range(20)],  # v. bad
            '21951':[i for i in range(20)],  # v. bad
            '21952':[i for i in range(20)],  # v. bad
            '21953':[i for i in range(20)],  # v. bad
            '21955':[i for i in range(20)],  # v. bad
            '21956':[i for i in range(20)],  # v. bad
            '21957':[i for i in range(20)],  # bad
            '21998':[i for i in range(20)],  # bad. 10 loss of sensitivity?
            '22001':[10],  # 10 loss of sensitivity?
            '22004':[10],  # 10 loss of sensitivity?
            '22006':[10],  # 10 loss of sensitivity?
            '22100':[i for i in range(20)],  # v. bad # 10 loss of sensitivity?
            '22129':[10],  # 10 loss of sensitivity?
            '22154':[10],  # 10 loss of sensitivity?
            '22209':[10],  # 10 loss of sensitivity? 10 is negative!!
            '22267':[10],  # 10 loss of sensitivity? 10 is negative!!
            '22296':[10],  # 10 loss of sensitivity? 10 is negative!!
            '22381':[10],  # 10 loss of sensitivity? 10 is negative!!
            '22410':[i for i in range(20)], # v. bad  # 10 loss of sensitivity? 10 is negative!!
            '23633':[i for i in range(20)],  # v. bad
            '24946':[i for i in range(20)],  # v. bad
            '25003':[1], # Spike appears directly ontop of Tau A in feed 1
            '25032':[i for i in range(20)], # Looks okay but all fluxes generally high? # 13,14,15, partially missing data?
            '25839':[i for i in range(20)],  # v. bad
            '25893':[i for i in range(20)],  # bad stripe?
            '26207':[i for i in range(20)],  # v. bad
            '26238':[i for i in range(20)],  # v. bad
            '26307':[i for i in range(20)],  # v. bad
            '26338':[i for i in range(20)],  # conjunction with saturn
            '26369':[i for i in range(20)],  # conjunction with saturn
            '26401':[i for i in range(20)],  # conjunction with saturn
            '26422':[i for i in range(20)],  # conjunction with saturn
            '26454':[i for i in range(20)],  # v. bad
            '26621':[i for i in range(20)],  # bad
            '27930':[i for i in range(20)],  # bad
            '28235':[i for i in range(20)],  # bad
            '28356':[i for i in range(20)],  # v. bad
            '28669':[i for i in range(20)],  # v. bad
            '28671':[i for i in range(20)],  # v. bad
            '28855':[i for i in range(20)],  # v. bad -- looks like the vane is in place???
            '28911-28965':[i for i in range(8,20)]+[3], # incorrect scanning strat to cover all feeds
            '29014':[i for i in range(20)],  # v. bad
            '29046':[i for i in range(20)],  # v. bad
            '29078':[i for i in range(20)],  # v. bad
            '29140':[i for i in range(20)],  # v. bad
            '29172':[i for i in range(20)],  # Looks fine but bad solution???
            '29203':[i for i in range(20)],  # Looks fine but bad solution???
            '29234':[i for i in range(20)],  # v. bad
            '29266':[i for i in range(20)],  # Looks fine but bad solution???
            '29297':[i for i in range(20)],  # Looks fine but bad solution???
            '29361':[i for i in range(20)],  # v. bad
            '29465':[i for i in range(20)],  # bad
            '29923':[i for i in range(20)],  # bad
            '29948':[i for i in range(20)],  # v. bad
            '29978':[i for i in range(20)],  # bad
            '30009':[i for i in range(20)],  # bad
            '30098':[i for i in range(20)]}  # v. bad

def calculate_cal_factors(data_dir):
    """Returns the calibration factors for each Tau A observation
    """

    FREQUENCIES = [27,29,31,33]
    filelist = glob.glob(f'{data_dir}/level3_*')

    amps   = np.zeros((len(filelist),20,4))
    offx   = np.zeros((len(filelist),20,4))
    offy   = np.zeros((len(filelist),20,4))
    obsids = np.zeros((len(filelist)))
    stdx   = np.zeros((len(filelist),20,4))
    stdy   = np.zeros((len(filelist),20,4))
    mjd    = np.zeros((len(filelist)))
    taua_model = CaliModels.TauAFluxModel()

    for ifile, filename in enumerate(tqdm(filelist)):
        h = h5py.File(filename,'r')
        if not 'FittedFluxes' in h:
            h.close()
            continue
        if not 'amplitude_0' in h['FittedFluxes']:
            h.close()
            continue
        amps[ifile] = h['FittedFluxes/amplitude_0'][...]
        offx[ifile] = h['FittedFluxes/x_mean_0'][...]
        offy[ifile] = h['FittedFluxes/y_mean_0'][...]
        stdx[ifile] = h['FittedFluxes/x_stddev_0'][...]
        stdy[ifile] = h['FittedFluxes/y_stddev_0'][...]
        mjd[ifile] = h['MJD'][0]

        obsids[ifile]= int(h['comap'].attrs['obsid'])

        #print(amps[ifile,0,0] * UnitConv.toJy(FREQUENCIES[0],1) * 2*np.pi * stdx[ifile,0]*stdy[ifile,0]*(np.pi/180.)**2/taua_model(FREQUENCIES[0],mjd[ifile]))
        h.close()

    # Mask bad observations
    mask = np.ones(amps.shape,dtype=bool)
    for k,v in badobs.items():
        if '<' in k:
            lim = int(k[1:])
            for ifeed in v:
                mask[obsids < lim,ifeed-1] = False
        elif '-' in k:
            low = int(k.split('-')[0])
            high= int(k.split('-')[1])
            for ifeed in v:
                mask[(obsids <= high) & (obsids >= low),ifeed-1] = False

        else:
            for ifeed in v:
                mask[obsids == int(k),ifeed-1] = False

    for iband in range(amps.shape[-1]):
        mask[(amps[:,:,iband] < 1),iband] = False
        mask[(mjd<1000),:,iband] = False
    amps[~mask] = np.nan


    
    for iband in range(amps.shape[-1]):
        amps[:,:,iband] = amps[:,:,iband] * UnitConv.toJy(FREQUENCIES[iband],1) * 2*np.pi * stdx[...,iband]*stdy[...,iband]*(np.pi/180.)**2/taua_model(FREQUENCIES[iband],mjd)[:,None]
    
    return obsids, amps

def apply_cal_factors(filename, cal_obsids, cal_factors):
    """
    """
    N_FEEDS = 20
    N_BANDS = 4

    h = h5py.File(filename,'a')
    obsid = int(h['comap'].attrs['obsid'])

    obs_cal_factors = np.zeros((N_FEEDS, N_BANDS))
    obs_cal_obsids  = np.zeros((N_FEEDS, N_BANDS))
    for ifeed in range(N_FEEDS):
        if ifeed == 19:
            continue
        for iband in range(N_BANDS):
            gd_sel = np.isfinite(cal_factors[:,ifeed,iband])
            gd_obs = cal_obsids[gd_sel]
            gd_cal = cal_factors[gd_sel,ifeed,iband]
            idx = np.argmin((gd_obs - obsid)**2)
            obs_cal_factors[ifeed,iband] = gd_cal[idx]
            obs_cal_obsids[ifeed,iband] = gd_obs[idx]

    grp_name = 'CALFACTORS'
    if grp_name in h:
        del h[grp_name]
    grp = h.create_group(grp_name)
    grp.create_dataset('calibration_factors',data= obs_cal_factors)
    grp.create_dataset('calibrator_obsids',data = obs_cal_obsids)
    
    h.close()

if __name__ == "__main__":

    
    LVL3_DIR = 'mock_level3_co2_update2'
    CAL_DIR  = 'mock_level3_TauA_update2/'
    filelist = glob.glob(f'{LVL3_DIR}/level3_*')

    cal_obsids, cal_factors = calculate_cal_factors(CAL_DIR)

    for filename in tqdm(filelist):
        try:
            apply_cal_factors(filename, cal_obsids, cal_factors)
        except KeyError:
            print(filename)
