# get_files.py <source> <output_file> --level <level>
#
# Produces a list of files for a given source that are available on the local system.
#

import numpy as np
from matplotlib import pyplot
from comancpipeline.Tools import FileTools
import os
import argparse

levels = {'level1':'',
          'level2':'_Level2Cont',
          'level2RRL':'_Level2RRL'}

parser = argparse.ArgumentParser()
parser.add_argument('source',
                    help='Name of the source.')
parser.add_argument('output_file',
                    help='File to write list of observations to.')
parser.add_argument('--level',default='level1',
                    help='What level of file are you searching for? Level1, Level2, Level2RRL')
args = parser.parse_args()

# Add information for remote server
presto_info = {'server':None,
               'directory':None,
               'script_location':None}

# Add local directories where level1/level2/level2rrl files are stored
local_info = {'level1':{'datadirs':None},
              'level2':{'datadirs':None},
              'level2RRL':{'datadirs':None}}

if __name__ == "__main__":
    source = args.source

    # Connect to remote server
    obsinfo = FileTools.query_sql(presto_info['server'],
                                  presto_info['script_location'],suffix=levels[args.level])
    sourceinfo = {}
    for obsid, target in obsinfo.items():
        if source.lower() in target.lower():
            sourceinfo[obsid] = target

    fout = open(args.output_file,'w')
    for k,v in sourceinfo.items():
        dataloc = [os.path.exists(f'{datadir}/{k}') for datadir in local_info[args.level]['datadirs']]
        if not any(dataloc):
            continue
        else:
            datadir = local_info[args.level]['datadirs'][dataloc][0]
        fout.write(f'{datadir}/{k}\n')
    fout.close()
