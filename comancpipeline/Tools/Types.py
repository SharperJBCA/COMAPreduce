import numpy as np

# Date Format from files:

DateFmt = '%Y-%m-%d-%H%M%S'

import datetime

def Filename2DateTime(filename):
    filename  = filename.split('/')[-1]
    try:
        date = datetime.datetime.strptime(filename[14:-4],DateFmt)
    except:
        date = datetime.datetime.strptime(filename[14:-4],DateFmt+'_Level2Cont')

    return date

def DateTime2Filename(date, obsID=0):
     filename = 'comap-{:05d}-{}.hd5'.format(obsID,datetime.datetime.strftime(DateFmt))

     return filename


#
_OTHER_       = -1
_HORNS_       = 0
_SIDEBANDS_   = 1
_FREQUENCY_   = 2
_TIME_        = 3

# DESCRIPTION OF COMAP DATA FILES TO ALLOW FOR MPI TO EFFICIENTLY SPLIT DATA

_COMAPDATA_ = {'spectrometer/tod': [_HORNS_,_SIDEBANDS_,_FREQUENCY_,_TIME_],
               'spectrometer/MJD': [_TIME_],
               'spectrometer/pixel_pointing/pixel_ra':[_HORNS_,_TIME_],
               'spectrometer/pixel_pointing/pixel_dec':[_HORNS_,_TIME_],
               'spectrometer/pixel_pointing/pixel_az':[_HORNS_,_TIME_],   
               'spectrometer/pixel_pointing/pixel_el':[_HORNS_,_TIME_],
               'spectrometer/frequency':[_SIDEBANDS_,_FREQUENCY_],
               'spectrometer/time_average':[_HORNS_,_SIDEBANDS_,_FREQUENCY_],
               'spectrometer/band_average':[_HORNS_,_SIDEBANDS_,_TIME_],
               'spectrometer/bands':[_SIDEBANDS_],
               'spectrometer/feeds':[_HORNS_]}

_SHAPECHANGES_ = {_HORNS_:False,
                  _SIDEBANDS_:False,
                  _FREQUENCY_:False,
                  _TIME_:False}

def getSplitStructure(splitdir, datastc=_COMAPDATA_):
    """
    splitdir, (integer)
    
    Search data structure description (datastc) to see if it contains data in a given direction (horns, time, etc...)

    Returns all fields that need splitting up
    """
    fields = {}

    if isinstance(splitdir, type(None)):
        return fields

    for k, v in datastc.items():
        if splitdir in v:
            fields[k] = np.where(np.array(v) == splitdir)[0][0]
    
    return fields

def getSelectStructure(selectdir, index, datastc=_COMAPDATA_):
    """
    selectdir, (integer)
    index, (integer) - index of axis to be selected

    e.g. You have an array described as :
    d = np.shape([_HORNS_,_FREQUENCY_])
    You want horns so selectdir = _HORNS_
    and you only want horn 0, therefore index = 0
    
    Search data structure description (datastc) to see if it contains data in a given direction (horns, time, etc...)

    Returns all fields in which only one index in selected
    """

    fields = {}

    if isinstance(selectdir, type(None)):
        return fields

    for k, v in datastc.items():
        if selectdir in v:
            fields[k] = [np.where(np.array(v) == selectdir)[0][0], index]

    return fields

def getFieldFullLength(svals, data, datastc=_COMAPDATA_):

    fields = {}
    for s in svals:
        for k, v in datastc.items():
            if s in v:
                selectAxis = np.where(np.array(v) == s)[0][0]
                if k in data:
                    fields[s] = data[k].shape[selectAxis]
                    break
    return fields
