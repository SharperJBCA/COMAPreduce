import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Ellipse

from scipy.interpolate import interp1d
from astropy.io import fits
from astropy import wcs

import h5py

from tqdm import tqdm 
import click
import ast
import os
 
from comancpipeline.Tools import ParserClass,binFuncs
from comancpipeline.MapMaking import DataReader, Destriper

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def bindata(x,y,nbins=10):

    xedges = np.linspace(np.min(x),np.max(x),nbins+1)
    xmids  = (xedges[1:]+xedges[:-1])/2.
    yvals = np.histogram(x,xedges,weights=y)[0]/np.histogram(x,xedges)[0]
    gd = np.isfinite(yvals)
    return xmids[gd], yvals[gd]

def build_covariance(offsets):

    # Clean out any zeros/gaps
    bd = (offsets == 0)
    x = np.arange(offsets.size)
    offsets[bd] = np.interp(x[bd],x[~bd], offsets[~bd])
    # ps = np.abs(np.fft.fft(offsets))**2
    # nu = np.fft.fftfreq(offsets.size,d=1.)
    
    # # build covariance
    # xb,yb = bindata(np.log10(nu[1:nu.size//2]),np.log10(ps[1:nu.size//2]),nbins=20)
    # ymdl = 10**np.interp(np.log10(nu[1:nu.size//2]),xb,yb)
    # cov  = np.abs(np.fft.ifft(ymdl))
    cov = np.correlate(offsets,offsets,mode='full')[:offsets.size]
    return cov[::-1]

def write_map(parameters,data,offsetMap,postfix=''):

    naive = data.naive.get_map()
    offmap= offsetMap.get_map()
    hits = data.naive.get_hits()
    variance = data.naive.get_cov()

    des = naive-offmap
    des[hits == 0] = np.nan
    clean_map = naive-offmap

    
    hdu = fits.PrimaryHDU(des,header=data.naive.wcs.to_header())
    cov = fits.ImageHDU(variance,name='Covariance',header=data.naive.wcs.to_header())
    hits = fits.ImageHDU(hits,name='Hits',header=data.naive.wcs.to_header())
    naive = fits.ImageHDU(naive,name='Naive',header=data.naive.wcs.to_header())

    hdul = fits.HDUList([hdu,cov,hits,naive])
    if not os.path.exists(parameters['Inputs']['maps_directory']):
        os.makedirs(parameters['Inputs']['maps_directory'])
    fname = '{}/{}_Feeds{}_Band{}{}.fits'.format(parameters['Inputs']['maps_directory'],
                                                      parameters['Inputs']['title'],
                                                      '-'.join([str(int(f)) for f in parameters['Inputs']['feeds']]),
                                                      int(parameters['ReadData']['iband']),postfix)
                                               
    hdul.writeto(fname,overwrite=True)

@click.command()
@click.argument('filename')#, help='Level 1 hdf5 file')
@click.option('--options', cls=PythonLiteralOption, default="{}")
def call_level1_destripe(filename, options):
    level1_destripe(filename, options)

def level1_destripe(filename,options):
    
    """Plot hit maps for feeds

    Arguments:

    filename: the name of the COMAP Level-1 file

    """
    # Get the inputs:
    p = ParserClass.Parser(filename)
    title = p['Inputs']['title'] 
    for k1,v1 in options.items():
        if len(options.keys()) == 0:
            break
        for k2, v2 in v1.items():
            p[k1][k2] = v2
    title = p['Inputs']['title']
    # Read in all the data
    if not isinstance(p['Inputs']['feeds'], list):
        p['Inputs']['feeds'] = [p['Inputs']['feeds']]
    filelist = np.loadtxt(p['Inputs']['filelist'],dtype=str,ndmin=1)

    
    np.random.seed(1)
    data = DataReader.ReadDataLevel2(filelist,
                                     feeds = p['Inputs']['feeds'],
                                     flag_spikes  =p['ReadData']['flag_spikes'],
                                     offset_length=p['Destriper']['offset'],
                                     ifeature     =p['ReadData']['ifeature'],
                                     feed_weights =p['Inputs']['feed_weights'],
                                     iband        =p['ReadData']['iband'],
                                     keeptod      =p['ReadData']['keeptod'],
                                     subtract_sky =p['ReadData']['subtract_sky'],
                                     map_info     =p['Destriper'])

    offsetMap, offsets = Destriper.Destriper(p, data)

    write_map(p,data,offsetMap,postfix='')

    ### 
    # Write out the offsets
    ###

    # ????

    ###
    # Write out the maps
    ###

if __name__ == "__main__":
    call_level1_destripe()
