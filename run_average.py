from mpi4py import MPI 
comm = MPI.COMM_WORLD

import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis import Calibration
from comancpipeline.Tools import Parser, Logging
import sys
import h5py
import os
from comancpipeline.Analysis.Calibration import NoHotError,NoColdError,NoDiodeError
import click

import ast
class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.argument('parameters')
@click.option('--classinfo' ,default='ClassParameters.ini',type=str)
@click.option('--start', default=None, type=int)
@click.option('--end', default=None, type=int)# cls=PythonLiteralOption, default="{}")

def call_main(parameters, classinfo, start, end):
    main(parameters, classinfo, start, end)

def main(parameters,classinfo, start=None, end=None):

    # Get the inputs:
    jobobjs, filelist, mainConfig, classConfig = Parser.parse_parameters(parameters)

    # Move this to parser and link to job objects?
    if 'LogFile' in mainConfig['Inputs']:
        logger = Logging.Logger(mainConfig['Inputs']['LogFile']+'_{}'.format(os.getpid()))
    else:
        logger = print

    if isinstance(start,type(None)):
        start = 0
    if isinstance(end, type(None)):
        end = len(filelist)

    # Execute object jobs:
    for filename in filelist[start:end]:
        print('Opening : {}'.format(filename))
        try:
            dh5 = h5py.File(filename, 'r')
        except OSError:
            print('Error: Could not open {}'.format(filename.split('/')[-1]))
            continue

        for job in jobobjs:
            try:
                dh5 = job(dh5)
            except (KeyError,NoHotError,NoColdError,NoDiodeError) as e:
                logger(filename,e)
                break
        dh5.close()

if __name__ == "__main__": 
    call_main()
