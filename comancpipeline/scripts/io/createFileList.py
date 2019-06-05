import os
import sys
import h5py
from os import listdir, getcwd
from os.path import isfile, join
import numpy as np

import optparse
default_parser=optparse.OptionParser(usage="Usage: %prog [options] filenames")
default_parser.add_option("-D","--filedir",dest='filedir',action="store",type='str',default='/scratch/nas_comap1/sharper/COMAP/data_ds/',help="Directory containing input files.")
default_parser.add_option("-o","--obstype",dest='obstype',action="store",type='str',default=None,help="Observation type to search for.")
default_parser.add_option("-V","--verbose",dest='verbose',action="store",default=False,help="")
default_parser.add_option("-F","--filelist",dest='filelist',action="store",type='str',default='filelist.list',help="Directory containing input files.")


if __name__ == "__main__":

    # Read in the metadata for this pipeline run
    options,pargs = default_parser.parse_args(sys.argv[1:])


    filelist = [f for f in listdir(options.filedir) if isfile(join(options.filedir, f)) if (('hdf5' in f) | ('hd5' in f))]

    outlist = []
    for f in filelist:
        try:
            d = h5py.File('{}/{}'.format(options.filedir,f),'r')
        except OSError:
            print('Could not open: {}'.format(f))
            continue

        if not 'comap' in d:
            continue
        if not 'comment' in d['comap'].attrs:
            continue

        comment = (d['comap'].attrs['comment'].lower()).decode('utf-8')
        if options.obstype.lower() in comment:
            if not 'test' in comment.lower():
                outlist += [f]
            if options.verbose:
                print(f)
        d.close()

    ab = np.zeros(len(outlist), dtype=[('filenames','U45')])
    ab['filenames'] = outlist
    np.savetxt(options.filelist, ab, fmt='%s')
