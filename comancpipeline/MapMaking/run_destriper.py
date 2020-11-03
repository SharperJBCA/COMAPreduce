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
from comancpipeline.MapMaking.Types import *
from comancpipeline.MapMaking.Destriper import Destriper, DestriperHPX

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


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
    parameters = ParserClass.Parser(filename)

    title = parameters['Inputs']['title'] 

    for k1,v1 in options.items():
        if len(options.keys()) == 0:
            break
        for k2, v2 in v1.items():
            parameters[k1][k2] = v2

    upperFrequency = parameters['Inputs']['upper_frequency']
    lowerFrequency = parameters['Inputs']['lower_frequency']
    title = parameters['Inputs']['title']

    # Read in all the data
    if not isinstance(parameters['Inputs']['feeds'], list):
        parameters['Inputs']['feeds'] = [parameters['Inputs']['feeds']]
    filelist = np.loadtxt(parameters['Inputs']['filelist'],dtype=str,ndmin=1)

    nside = int(parameters['Inputs']['nside'])
    data = DataLevel2AverageHPX(filelist,parameters,nside=nside,keeptod=True,subtract_sky=False)
    
    offsetMap, offsets = DestriperHPX(parameters, data)

    ###
    # Write out the offsets
    ###

    # ????

    ###
    # Write out the maps
    ###
    naive = data.naive()
    offmap= offsetMap()
    hits = data.hits.return_hpx_hits()
    variance = data.naive.return_hpx_variance()

    des = naive-offmap
    des[des == 0] = hp.UNSEEN
    naive[naive == 0] = hp.UNSEEN
    variance[variance == 0] = hp.UNSEEN
    offmap[offmap==0] = hp.UNSEEN
    hits[hits == 0] = hp.UNSEEN
    clean_map = naive-offmap
    clean_map[clean_map==0]=hp.UNSEEN


    if 'maps_directory' in parameters['Inputs']:
        map_dir = parameters['Inputs']['maps_directory']
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
    else:
        map_dir = '.'

    
    hp.write_map('{}/{}_{}-{}.fits'.format(map_dir,title,upperFrequency,lowerFrequency), [clean_map, variance, naive, offmap,hits],overwrite=True,partial=True)

    feedstrs = [str(v) for v in parameters['Inputs']['feeds']]


if __name__ == "__main__":
    call_level1_destripe()
