# assign_stats_flags.py
#
# Apply flagged state to each obsid in comap_database based on user defined
# cuts on the fitted noise statistics and other stats
#
 
import numpy as np
from matplotlib import pyplot
import h5py
from tqdm import tqdm
import os
import pickle

def flag_obs(grp,flagged=True, feeds='all'):
    """
    Creates required attributes with default parameters
    """
    grp.attrs['Flagged'] = flagged
    grp.attrs['Flagged_Feeds'] = feeds

def flag_common(grp,name,flagged=True):
    grp.attrs[name] = flagged

def flag_basics(grp):
    names = ['Vane','level2','FnoiseStats','FitSource']
    for name in names:
        flag_common(grp,name,flagged=True)

def create_flags(filename,max_rednoise=3e-3,min_moondist=1, min_sundist=7.5,min_alpha=-1.5):
    """
    Create flags based on noise stats and assign to each obsid
    """
    flags = {'obsidcut':{'dataset':None,
                         'limit': lambda x: ((x >= 7523) & (x <= 7554)) | (x == 13823)},
             'sun':{'dataset':'SunDistance/sun_mean',
                    'limit':lambda x: x < min_sundist, 
                    'slice':(slice(None)),
                    'process':lambda x: np.nanmean(x)},
             'moon':{'dataset':'SunDistance/moon_mean',
                     'limit':lambda x: x < min_moondist, 
                     'slice':(slice(None)),
                     'process':lambda x: np.nanmean(x)},
             'rednoise':{'dataset':'FnoiseStats/fnoise_fits',
                         'limit':lambda x: x > max_rednoise, 
                         'slice':(slice(0,1),slice(2,3),slice(0,32),slice(0,1),slice(0,1)),
                         'process':lambda x: np.nanmedian(x)},
             'alpha':{'dataset':'FnoiseStats/fnoise_fits',
                         'limit':lambda x: x < min_alpha, 
                         'slice':(slice(0,1),slice(2,3),slice(0,32),slice(0,1),slice(1,2)),
                         'process':lambda x: np.nanmedian(x)},
             'channelcount':{'dataset':'FnoiseStats/fnoise_fits',
                             'limit':lambda x: x > 50, 
                             'slice':(slice(0,1),slice(None),slice(None),slice(0,1),slice(0,1)),
                             'process':lambda x: x.size-np.nansum(np.isfinite(x.flatten()) & (x.flatten() != 0))}}
    count = 0
    h = h5py.File(filename,'a')

    el = []

    for i,(obsid, grp) in enumerate(tqdm(h.items())):
        flag_basics(grp) # setup some basic flags
        if not 'Vane' in grp:
            flag_obs(grp)
            flag_common(grp,'Vane',False)
            continue
        if not 'level2' in grp:
            flag_obs(grp)
            flag_common(grp,'level2',False)
            continue
        if not 'CompareTsys' in grp:
            flag_obs(grp)
            continue
        if not 'SunDistance' in grp:
            flag_obs(grp)
            continue
        if not 'FnoiseStats' in grp:
            flag_obs(grp)
            flag_common(grp,'FnoiseStats',False)
            continue

        source = grp['level2'].attrs['source'].split(',')[0].strip()
        if source == 'TauA':
            if not 'FitSource' in grp:
                flag_common(grp,'FitSource',False)
                flag_obs(grp)
                continue
        if source in grp['level2'].attrs['source']:
            if grp['FnoiseStats/fnoise_fits'].shape[-2] == 0: # Catch very old 1/f noise fits
                flag_common(grp,'FnoiseStats',False)
                flag_obs(grp)
                continue
            test = grp['FnoiseStats/fnoise_fits'][0,0,:,0,1] # Catch old 1/f noise fits
            testval = np.nanmedian(test[test != 0])
            if testval > 0:
                flag_common(grp,'FnoiseStats',False)
                flag_obs(grp)
                continue

            flagtests = []
            for flagname, flag in flags.items():
                if isinstance(flag['dataset'],type(None)):
                    flagtests += [flag['limit'](int(obsid))]
                else:
                    d = grp[flag['dataset']][flag['slice']]
                    stat = flag['process'](d)
                    flagtests += [flag['limit'](stat)]
            if any(flagtests):
                flag_obs(grp)
            else:
                flag_obs(grp,flagged=False)

            # Store what flag the data failed on
            for (name,flag) in zip(list(flags.keys()), flagtests):
                flag_common(grp,name,flag)
            count += 1
    h.close()


if __name__ == "__main__":

    create_flags('comap_database.hdf5')
