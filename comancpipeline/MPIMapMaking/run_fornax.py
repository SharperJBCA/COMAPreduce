# run_jackknife.py
#
# Runs a file list and produces a set of jack knives to look at
#

import numpy as np
from comancpipeline.Tools import Coordinates
import jackknife_funcs
import os
import subprocess
import time

def ParameterFile(filelist,
                  offset_length=50,
                  prefix='map',
                  output_dir='maps/',
                  feeds =[1],
                  nxpix = 480,
                  nypix = 480,
                  crval = [0,0],
                  crpix = [240,240],
                  ctype = ['RA---CAR','DEC--CAR'],
                  cdelt = [-1./60.,1./60.]):

    out = """[Inputs]
    [Inputs]

    filelistname= {filelist}
    offset_length= {offset_length}
    prefix= {prefix}
    output_dir= {output_dir}
    feeds= {feeds}
    nxpix= {nxpix}
    nypix= {nypix}
    crval=  {crval}
    crpix= {crpix}
    ctype= {ctype}
    cdelt= {cdelt}
    """.format(filelist=filelist,
               offset_length=offset_length,
               prefix=prefix,
               output_dir=output_dir,
               feeds = ','.join([str(f) for f in feeds]),
               nxpix = nxpix,
               nypix = nypix,
               crval = ','.join([str(s) for s in crval]),
               crpix = ','.join([str(s) for s in crpix]),
               ctype = ','.join([str(s) for s in ctype]),
               cdelt = ','.join([str(s) for s in cdelt]))


    fout = open(parameter_filename,'w')
    fout.write(out)
    fout.close()
    return out

def write_pbs_script(name, email, work_dir, script):
    pbs_script = """#!/bin/bash
    #PBS -l nodes=1:ppn=32

    # Name of Job 
    #PBS -N {name}

    # Logs
    #PBS -j oe
    
    # Email
    #PBS -m ae
    #PBS -M {email}
    
    #PBS -V
    PBS_O_WORKDIR={work_dir}
    cd $PBS_O_WORKDIR
    echo "Running job in" $PBS_O_WORKDIR
    pwd
    
    echo "list of nodes being used";
    cat $PBS_NODEFILE

    num_procs=$(cat $PBS_NODEFILE | wc -l)
    {script}
    """.format(name=name, email=email, work_dir=work_dir, script=script)

    script_name = 'pbs_dir/{}_pbs.script'.format(name)
    fout = open(script_name,'w')
    fout.write(pbs_script)
    fout.close()
    return script_name


    info = {'names':['fg4',,'fg7','fg9','fg10','fg11'],
            'filelists':['Filelists/fg4.txt',
                         'Filelists/baddests_list_fg7.txt'
                         'Filelists/baddests_list_fg9.txt',
                         'Filelists/baddests_list_fg10.txt',
                         'Filelists/baddests_list_fg11.txt'],
            'pipeline_str':','.join(['Calibration.CreateLevel2Cont',
                                     'Statistics.ScanEdges',
                                     'Statistics.SunDistance',
                                     'CreateLevel3.CreateLevel3',
                                     'CreateLevel3.Level3FnoiseStats']),
            'parameter_filename':'ParameterFiles/{}_parameters.ini'}

    email = 'stuart.harper@manchester.ac.uk'
    work_dir = '/scratch/nas_comap1/sharper/COMAP/comap_pipeline_2022/data_reduction'
    script = 'mpirun -n 16 python run_destriper.py {}'

    for (name,filelist) in zip(info['names'],info['filelists']):
        parameter_filename = info['parameter_filename'].format(name)
        ParameterFile(parameter_filename,
                      info['pipeline_str'],
                      filelist)

        pbs_script = write_pbs_script(name, email,
                                      work_dir, 
                                      script.format(parameter_filename))

        subprocess.run(['qsub',pbs_script])

