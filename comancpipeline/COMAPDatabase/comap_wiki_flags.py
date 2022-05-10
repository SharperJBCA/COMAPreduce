"""
download_flags.py

Sets whether an observation is 'flagged' or not in the comap_database. 

Flags are taken from the COMAP observer flags and a set of fixed cuts. 

Observer flags can be per feed, while cuts currently are not per feed.

Flags are used during Level3 creation and map making.

To run: python download_flags.py 

Version 1: 29/03/22 - SH NB: need to add an option for whether to update the local csv or not

"""

import requests
import gspread
import csv
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from tqdm import tqdm
def update_csv():
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json',scope)

    docid = '1ab23NlqiUetoygd6PWlbmwtgF70V1WcOhUhFo5XtSWo'#COMAP observer flags'
    
    client = gspread.service_account(filename='credentials.json')#credentials)    
    spreadsheet = client.open_by_key(docid)

    for i, worksheet in enumerate(spreadsheet.worksheets()):
        filename = docid + '-worksheet' + str(i) + '.csv'
        with open(filename,'w') as f:
            writer = csv.writer(f)
            writer.writerows(worksheet.get_all_values())

def parse_flags(filename):
    """
    Parser the flag database
    """

    f = open(filename,'r')

    rows = []
    csvreader = csv.reader(f,delimiter=',')
    iline = 0
    for line in csvreader:
        if iline > 0:
            
            idstart,idend,pixels,sidebands,reason,author,date = line#.split(',')
            idstart = int(idstart)
            if len(idend) > 0:
                idend = int(idend)
            else:
                idend = None
            if pixels.lower() == 'many':
                pixels = 'all'
            if pixels.lower() != 'all':
                pixels = [int(p) for p in pixels.split(',')]
                
            rows += [[idstart,idend, pixels]]
        iline += 1
    f.close()
    return rows

def check_id(rows,obsid):
    """
    Check an obsid
    """

    for row in rows:
        start,end,pixels = row
        if isinstance(end,type(None)):
            if start == obsid:
                return True, pixels
        else:
            if (obsid >= start) & (obsid <= end):
                return True, pixels
    return False, None

import h5py

def check_cuts(grp):
    """
    See if this obs id needs to be cut
    """

    min_sun_distance = 7 # degrees
    max_fnoise_at_10s= 0.15 # K^2
    max_feed_covariance= 0.01 # K^2
    max_beam_sigx = 6/60./2.355 # degrees

    ref_freq = 0.1 # Hz
    fits   = grp['FnoiseStats']['fnoise_fits'][...]
    fnoise = fits[0,...,0,0]**2*(1 + (ref_freq/10**fits[0,...,0,1])**fits[0,...,0,2])
    fnoise = np.nanmedian(fnoise)
    cuts = [grp['SunDistance']['sun_mean'][...] < min_sun_distance,
            grp['FeedFeedCorrelations']['feed_feed_correlation'][...] > max_feed_covariance,
            fnoise > max_fnoise_at_10s]

    if 'FitSource' in grp:
        cuts += [grp['FitSource']['Avg_Values_Band0'][2] > max_beam_sigx]

    if any(cuts):
        return True
    else:
        return False

if __name__ == "__main__":

    if params['update_csv']:
        update_csv()

    rows = parse_flags('1ab23NlqiUetoygd6PWlbmwtgF70V1WcOhUhFo5XtSWo-worksheet0.csv')
    h = h5py.File('comap_database.hdf5','a')

    for obsid, grp in tqdm(h.items()):
        flagged,feeds = check_id(rows,int(obsid))
        if flagged:
            grp.attrs['Flagged'] = True
            grp.attrs['Flagged_Feeds'] = feeds
        else:
            grp.attrs['Flagged'] = False
            grp.attrs['Flagged_Feeds'] = []
    h.close()
