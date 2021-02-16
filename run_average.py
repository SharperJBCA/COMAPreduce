from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
    jobobjs, filelist, mainConfig, classConfig, logger = Parser.parse_parameters(parameters)

    # Move this to parser and link to job objects?

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
            if rank == 0:
                print(job,flush=True)
            try:
                dh5 = job(dh5)
            except Exception as e: 
                fname = filename.split('/')[-1]
                logger(f'{fname}:{e}',error=e)
                break
            if isinstance(dh5, type(None)): # Return None means skip
                break
        if not isinstance(dh5, type(None)): 
            dh5.close()
        logger('############')
        

if __name__ == "__main__": 
    call_main()
