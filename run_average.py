import matplotlib
matplotlib.use('Agg')

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

    # Pass to rank 0
    if rank == 0:
        pids = np.zeros(size,dtype=int)
        pids[0] = os.getpid()
    else:
        pids = None
    for inode in range(1,size):
        if rank == inode:
            pid = np.array([os.getpid()])
            comm.Send(pid,dest=0,tag=inode)
        if rank == 0:
            pid = np.zeros(1,dtype=int)
            comm.Recv(pid, source=inode, tag=inode)
            pids[inode] = pid[0]
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

    if rank == 0:
        database = mainConfig['Inputs']['database']
        db = h5py.File(database,'a')
        for pid in pids:
            if not os.path.exists(f'{database}_{pid}'):
                continue
            db_pid = h5py.File(f'{database}_{pid}','r')
            for k,v in db_pid.items():
                if k in db:
                    for k2,v2 in v.items():
                        if k2 in db[k]:
                            del db[k][k2]
                        db_pid[k].copy(k2,db[k])
                else:
                    db_pid.copy(k,db)
            db_pid.close()
            os.remove(f'{database}_{pid}')


if __name__ == "__main__": 
    call_main()
