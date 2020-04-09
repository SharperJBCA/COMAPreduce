from mpi4py import MPI 
comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser
import sys
import h5py
import os

import click
@click.command()
@click.argument('parameters')
@click.option('--classinfo' ,default='ClassParameters.ini',type=str)
def main(parameters,classinfo):

    # Get the inputs:
    jobobjs, filelist, mainConfig, classConfig = Parser.parse_parameters(parameters)


    # Execute object jobs:
    for filename in filelist:
        print('Opening : {}'.format(filename))
        dh5 = h5py.File(filename, 'r')
        for job in jobobjs:
            dh5 = job(dh5)
        dh5.close()

if __name__ == "__main__": 
    main()
