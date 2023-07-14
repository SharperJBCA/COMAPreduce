import h5py 
import numpy as np
from matplotlib import pyplot
import os
import sys 
import healpy as hp 
from tqdm import tqdm
import glob 
from CBASSData import CBASSData

if __name__ == "__main__":

    quicksavefile = 'tmp_files/quick_save_eastwest.npy'

    if not os.path.exists(quicksavefile):
        filelist = glob.glob(f'{sys.argv[1]}/*.h5')
        az = []
        ra = []
        el = []
        dec = []
        tod = []
        for filename in tqdm(filelist):

            h = h5py.File(filename,'r')
            flags = h['flag'][...]
            special = CBASSData.calc_empty_offsets(flags, offset_length=500)
            az.append(h['az'][special])
            ra.append(h['ra'][special])
            el.append(h['el'][special])
            dec.append(h['dec'][special])
            t = (h['I1'][...]*h['wI1'][...] + h['I2'][...]*h['wI2'][...])/(h['wI1'][...]+h['wI2'][...])
            mask = (flags == 0)
            t = t[special]-h['offset'][...]
            t -= np.nanmedian(t)

            tod.append(t)
            h.close()

        tod = np.concatenate(tod)
        az = np.concatenate(az)
        ra = np.mod(np.concatenate(ra)*180/np.pi,360)
        el = np.concatenate(el)
        dec = np.concatenate(dec)*180/np.pi
        np.save(quicksavefile,[tod,az,ra,el,dec])
    else:
        tod,az,ra,el,dec = np.load(quicksavefile)

    nbins = 128
    ra_edges = np.linspace(0,360,nbins+1)
    az_edges = np.linspace(0,360,nbins+1)

    gd = np.isfinite(tod)
    hist2d = np.histogram2d(ra[gd],az[gd],bins=[ra_edges,az_edges],weights=tod[gd])[0]
    hist2d_counts = np.histogram2d(ra[gd],az[gd],bins=[ra_edges,az_edges])[0]

    hist2d_image = hist2d/hist2d_counts

    pyplot.imshow(hist2d_image,extent=[0,360,0,360],origin='lower',vmax=0.1)
    pyplot.colorbar()
    pyplot.savefig('tod.png')
    pyplot.close()

    pyplot.imshow(hist2d_counts,extent=[0,360,0,360],origin='lower')#,vmax=0.1)
    pyplot.colorbar()
    pyplot.savefig('counts.png')
    pyplot.close()

    hist2d_image = hist2d/hist2d_counts
    pyplot.imshow(hist2d_image-hist2d_image[:,::-1],extent=[0,360,0,360],origin='lower',vmax=0.025,vmin=-0.025)
    pyplot.colorbar()
    pyplot.savefig('tod_diff.png')
    pyplot.close()

    # diff by az east/west split
    east = hist2d_image[:,nbins//2:]
    west = hist2d_image[:,:nbins//2]
    diff = east-west
    for i in range(diff.shape[0]):
        pyplot.plot(diff[i,:],'k',alpha=0.1)
    pyplot.plot(np.mean(diff,axis=0),'C3',lw=3)
    pyplot.ylim(-0.025,0.025)
    pyplot.savefig('diff.png')
    pyplot.close()