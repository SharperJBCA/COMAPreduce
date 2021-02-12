import numpy as np
from matplotlib import pyplot
import glob
import h5py
from tqdm import tqdm
import sys
import os
if __name__ == "__main__":

    if not os.path.exists('FileLists'):
        os.makedirs('FileLists')

    datadir = '/scratch/nas_comap2/sharper/COMAP/level2/BW16/'
    filelist = np.sort(glob.glob(datadir+'/*.hd5'))
    source = sys.argv[1]

    cutoff = 4e-3 # K
    fout2 = open(f'FileLists/{source}_level2.list','w')
    fout1 = open(f'FileLists/{source}_level1.list','w')
    fout3 = open(f'FileLists/{source}_level2_rejected.list','w')

    for ifile,f in enumerate(tqdm(filelist[:])):
        try:
            h = h5py.File(f,'r')
        except:
            print(f)
            continue

        try:
            src = h['level1/comap'].attrs['source'].decode('ascii')
            comment = h['level1/comap'].attrs['comment'].decode('ascii')
        except KeyError:
            h.close()
            continue

        #print(ifile,src,z,f)

        if (source in src) & (not 'sky' in comment):
            # print(h.keys())
            # if not 'Statistics' in h['level2']:
            #     continue
            f1 = h.filename.split('/')[-1]
            f1 = f1.split('_Level2')[0]+'.hd5'
            fout1.write('/scratch/nas_comap2/sharper/COMAP/data/{}\n'.format(f1))            
            if not 'level3' in h:
                continue

            stats = h['level2/Statistics']
            fnoise = stats['fnoise_fits'][...]
            wnoise = stats['wnoise_auto'][...]
            try:
                sig_f = wnoise[0,...,0,0]**2 * (1. + (1./fnoise[0,...,0,0])**fnoise[0,...,0,1])
            except IndexError:
                continue
            if np.nanmedian(sig_f) < cutoff:
                fout2.write('{}\n'.format(f)) 
            else:
                fout3.write('{}\n'.format(f)) 

        h.close()
    fout1.close()
    fout2.close()
    fout3.close()
