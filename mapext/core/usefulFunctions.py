import h5py
import os
import numpy as np

import matplotlib.pyplot as plt

def h5PullDict(obj):
    out = {}
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                out[key] = h5PullDict(value)
            else:
                out[key] = np.asarray(value)
    return out

def h5PushDict(obj, dic, ow=True):
    if isinstance(obj, h5py.Group):
        for key, item in dic.items():
            if isinstance(item, dict):
                if key not in obj:
                    subgrp = obj.create_group(key)
                h5PushDict(obj[key], item)
            else:
                if (key in obj) and (ow):
                    del obj[key]
                try:
                    if item != None:
                        obj[key] = item
                except:
                    obj[key] = item

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def set_lims_inplace(data,max_step=1,lower=0.05,upper=None,delta=False,interest='all',maxvalcap=np.inf,minvalcap=-np.inf,sym=False):
    if upper == None:
        upper = 1-lower
    mask = np.all([np.isfinite(data),data!=0],axis=0)
    if interest=='all':
        min = np.nanmin(data[mask])
        max = np.nanmax(data[mask])
    else:
        min = interest[0]
        max = interest[1]
    if max>maxvalcap:
        max = maxvalcap
    if min>maxvalcap:
        min = maxvalcap
    if min<minvalcap:
        min = minvalcap
    if max<minvalcap:
        max = minvalcap
    step = (max-min)/2000
    if step>max_step:
        step=max_step
    print (min,max)
    hist,edge = np.histogram(data[mask], bins=np.arange(min,
                                                        max+1e-26,
                                                        step))
    cen = 0.5*(edge[1:]+edge[:-1])
    hist = np.cumsum(hist)
    if delta:
        hist = hist-hist[0]
    hist = hist/np.nanmax(hist)
    lims = np.interp([lower,upper],hist,cen)
    if sym:
        upper = np.max(np.abs([lower,upper]))
        lower = -upper
    return {'vmin':lims[0],'vmax':lims[1]}

def set_lims_inplace_hist2d(data,max_step=1e-3,lower=0.05,upper=None,delta=False,interest='all',maxvalcap=10,minvalcap=-10,sym=False):
        if upper == None:
            upper = 1-lower
        mask = np.all([np.isfinite(data),data!=0],axis=0)
        if interest=='all':
            min = np.nanmin(data[mask])
            max = np.nanmax(data[mask])
        else:
            min = interest[0]
            max = interest[1]
        if max>maxvalcap:
            max = maxvalcap
        if min>maxvalcap:
            min = maxvalcap
        if min<minvalcap:
            min = minvalcap
        if max<minvalcap:
            max = minvalcap
        step = (max-min)/2000
        if step>max_step:
            step=max_step
        print(min,max)
        hist,edge = np.histogram(data[mask], bins=np.arange(min,
                                                            max+1e-26,
                                                            step))
        cen = 0.5*(edge[1:]+edge[:-1])
        hist = np.cumsum(hist)
        if delta:
            hist = hist-hist[0]
        hist = hist/np.nanmax(hist)
        lims = np.interp([lower,upper],hist,cen)
        if sym:
            upper = np.max(np.abs([lower,upper]))
            lower = -upper
        return lims
