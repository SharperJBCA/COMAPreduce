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
if __name__ == "__main__":

    #x0 = Coordinates.sex2deg('00:40:00.095',hours=True)
    #y0 = Coordinates.sex2deg('+40:59:41.73')

    source_name = 'fg4'
    email = 'nothing'# 'stuart.harper@manchester.ac.uk'
    work_dir = '/scratch/nas_comap1/sharper/COMAP/comap_pipeline_2022/map_making'
    # 1) Split the filelists in to groups with output directories
    filename = 'Filelists/{}_good.txt'.format(source_name)
    filelist_info = jackknife_funcs.split_filelists(filename)

    # 2) Execute the map-making procedure on each node of the fornax cluster
    x0 = 55.
    y0 = 31.8
    x0 = Coordinates.sex2deg('00:40:00.095',hours=True)
    y0 = Coordinates.sex2deg('+40:59:41.73')

    map_info = dict(nypix=450,
                    nxpix=450,
                    cdelt='{},{}'.format(-1./60.,1./60.),
                    crpix='{},{}'.format(225,225),
                    crval='{},{}'.format(x0,
                                         y0),
                    ctype='{},{}'.format('RA---TAN','DEC--TAN'))
    parameter_files = jackknife_funcs.create_parameter_files(source_name, 
                                                             filelist_info, map_info)

    for name, script in parameter_files.items():
        filename = jackknife_funcs.write_pbs_script(name, email, work_dir, script)
        subprocess.run(['qsub',filename])
        time.sleep(5)
        #stop
    ### In a separate function
        # 3) Combine together each of the maps combinations into residual maps

    # 4) Produce figures and statistics of each map residual pairing.
