# query_source.py
#
# Produce filelists for a particular source that include:
# - Good observations that have been processed up to at least level 2 [{source}.dat]
# - Observations that need further processing [{pipeline}_{source}.dat]
# - Bad observations that failed stats or date range tests [bad_{source}.dat]

import numpy as np
from matplotlib import pyplot
import os
import h5py
from tqdm import tqdm

def check_reason(obs):
    names = ['Vane','level2','FnoiseStats','FitSource']
    reason = 'bad'
    for name in names:
        if obs.attrs[name] == False:
            reason = name
    return reason

def check_obs(obs):
    
    if obs.attrs['Flagged']:
        reason = check_reason(obs)
    else:
        reason = 'good'

    return reason

def query_source(filename,source,datadir='outfiles/'):
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    bad_files = '{}/badfiles.dat'.format(datadir)

    bad_files_unt = open(bad_files,'w')

    outfiles = {}

    h = h5py.File(filename,'r')

    for obsid, obs in tqdm(h.items()):
        if not 'level2' in obs:
            bad_files_unt.write('{}\n'.format(obsid))
            continue
        if not source in obs['level2'].attrs['source']:
            continue
        reason = check_obs(obs)
        if not 'level2_filename' in obs.attrs:
            reason = 'level1_'+reason
            filename = 'level1_filename'
        else:
            filename = 'level2_filename'
        if not reason in outfiles:
            outfiles[reason] = open('{}/{}_{}.dat'.format(datadir,
                                                          reason,
                                                          source),'w')

        outfiles[reason].write('{}\n'.format(obs.attrs[filename]))

    h.close()

import sys
if __name__ == "__main__":
    
    query_source('comap_database.hdf5',sys.argv[1])
