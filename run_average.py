from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np
#from comancpipeline.Tools import *
from comancpipeline.Tools import Parser, Logging
import sys
import h5py
import os
import click
from tqdm import tqdm

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
    jobobjs, prejobobjs, filelist, mainConfig, classConfig, logger = Parser.parse_parameters(parameters)
    logger(f'STARTING')

    # Only let rank 0 do prejobs
    if rank == 0:
        for job in prejobobjs:
            job()

    comm.Barrier()

    # Move this to parser and link to job objects?

    if isinstance(start,type(None)):
        #start = 0
        #end = len(filelist)
        idx=np.sort(np.mod(np.arange(len(filelist)),size))
        select = np.where((idx == rank))[0]
        start = select[0]
        end   = select[-1] + 1
    # Execute object jobs:
    for filename in tqdm(filelist[start:end]):
        try:
            dh5 = h5py.File(filename, 'r')
        except OSError as e:
            logger(f'{filename}:{e}',error=e)
            continue

        for job in jobobjs:
            if rank == 0:
                print(job,flush=True)
            try:
            #if True:
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
