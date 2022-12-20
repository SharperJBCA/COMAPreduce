import numpy as np
import matplotlib
matplotlib.use('Agg')
import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from matplotlib import pyplot
from comancpipeline.Analysis import GainCorrection, Calibration
import sys
import comancpipeline

from collections import Counter
import linecache
import os
import tracemalloc

def logger_dummy(*args,**kwargs):
    pass

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


if __name__ == "__main__":

    output_dirs = ['/mn/stornext/d22/cmbco/comap/continuum/continuum_pipeline/mock_level3_co2_update2/']
    database = '/mn/stornext/d22/cmbco/comap/continuum/continuum_pipeline/cont_database.hd5'
    pipeline = [Calibration.CalculateVaneMeasurement(calvanedir = 'CalVanes',
                                                     overwrite=False,
                                                     output_obsid_starts = [0],
                                                     output_obsid_ends   = [None],
                                                     output_dirs = output_dirs,
                                                     database   = database,
                                                     elLim=5, 
                                                     feeds = 'all',
                                                     minSamples=200,
                                                     tCold=2.73,
                                                     set_permissions=False,
                                                     permissions_group='comap',
                                                     tHotOffset=273.15,
                                                     logger=logger_dummy,
                                                     vaneprefix='VaneCal'),
                GainCorrection.CreateLevel2GainCorr(feeds = 'all',
                                                    overwrite = True,
                                                    output_obsid_starts = [0],
                                                    output_obsid_ends   = [None],
                                                    output_dirs = output_dirs,
                                                    database = database,
                                                    permission_group = 'astcomap',
                                                    set_permissions = False,
                                                    average_width = 16,
                                                    calvanedir = 'CalVanes',
                                                    logger=logger_dummy,
                                                    calvane_prefix = 'VaneCal')]


    filelist = np.loadtxt(sys.argv[1],dtype=str,ndmin=1)
    #l1_dir = '/mn/stornext/d22/cmbco/comap/protodir/level1/'
    #file_list = ['2021-07/comap-0022001-2021-07-15-142002.hd5', '2021-12/comap-0025911-2021-12-10-082530.hd5']
    #filelist = np.array([l1_dir+f for f in file_list])
    idx=np.sort(np.mod(np.arange(len(filelist)),size))
    select = np.where((idx == rank))[0]
    tracemalloc.start()
    for filename in filelist[select]:
        h = h5py.File(filename,'a')
        #print(filename)
        #print(type(h))
        for job in pipeline:
            #print(type(h))
            try:
                h = job(h)
            except (comancpipeline.Analysis.Calibration.NoHotError,comancpipeline.Analysis.Calibration.NoDiodeError,KeyError):
                #,IndexError,TypeError,ValueError):
                break
            #snapshot=tracemalloc.take_snapshot()
            #display_top(snapshot)
            if isinstance(h,type(None)):
                break
        h.close()
        
    print(f'RANK {rank} FINISHED')
