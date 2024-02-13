import matplotlib
matplotlib.use('agg')

import os
import toml

from comancpipeline import Analysis 
from comancpipeline.Analysis.Running import set_logging

import glob 
#from comancpipeline.MapMaking.run_destriper import main as destriper_main 

from mpi4py import MPI 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np 

def create_tod_processing(configuration): 
    
    #filelist_name, figure_directory='figures', level2_directory='level2'):
    """This is the tod processing loop"""

    figure_directory = configuration['Global']['level2_figures'] 
    filelist_name = configuration['Global']['level1_filelist']
    level2_directory = configuration['Global']['level2_data']
    
    if rank == 0:
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)
    if rank == 0:
        if not os.path.exists(level2_directory):
            os.makedirs(level2_directory)
    comm.Barrier() 

    filelist =np.loadtxt(filelist_name,dtype=str, ndmin=1)
    
    idx = np.sort(np.mod(np.arange(filelist.size),size)) 
    idx = np.where((idx == rank))[0] 
    
    tod_processing = Analysis.Runner()

    processes={Analysis.CheckLevel1File: {'overwrite': True},
               Analysis.AssignLevel1Data: {'overwrite': False,'write':True}} 
    
    for process_name in configuration['Global']['processes']:
        processes[getattr(Analysis,process_name)] = configuration[process_name]
        processes[getattr(Analysis,process_name)]['figure_directory'] = figure_directory

    tod_processing.level2_data_dir = level2_directory
    tod_processing.filelist  = filelist[idx]
    tod_processing.processes = processes

    return tod_processing 

def create_astro_cal(targets, cal_files, source='TauA', level2_directory='level2',figure_directory='figures'):
    """ 
    1) Assign dates to the target files and cal_files
    1a) If check_calibrators:  
    2) Loop through each target file and assign nearest 'good' calibration observation
    3) Apply the nearest calibrator to the target file
    """
    filelist = np.loadtxt(targets,dtype=str, ndmin=1)
    cal_filelist = np.loadtxt(cal_files,dtype=str, ndmin=1)
        
    cal_processing = Runner()
    processes = {ApplyCalibration:{'calibrator_filelist':cal_filelist,'overwrite_calibrator_file':False,'calibrator_source':source,
                                   'figure_directory':figure_directory,'nowrite':False}}

    cal_processing.level2_data_dir = level2_directory
    cal_processing.filelist  = filelist
    cal_processing.processes = processes

    return cal_processing 

def map_making(targets, source='TauA'):
    """ 
    """
    parameters = dict(offset_length=int(50),
                    prefix='fg9',
                    output_dir='/scratch/nas_core/sharper/COMAP/maps/fg9/',
                    feeds=[i+1 for i in range(19)],
                    feed_weights=None,
                    nxpix=int(512),
                    nypix=int(512),
                    crval=['05:32:00.3','+12:30:28'],
                    crpix=[256,256],
                    ctype=['RA---SIN','DEC--SIN'],
                    cdelt=[-1./60.,1./60.])

    destriper_main(targets, **parameters) 

def create_plot_processing(level2_data_dir, source, output_dir):
    """ 
    """
    plot_processing = Runner()
    processes = {Level2Timelines:{'figure_directory':output_dir, 'source':source}}

    plot_processing.level2_data_dir = level2_data_dir
    plot_processing.filelist  = glob.glob(f'{level2_data_dir}/*.hd5')
    plot_processing.processes = processes

    return plot_processing

def main():
    import sys
    if True:

        toml_filename = sys.argv[1] 
        with open(toml_filename,'r') as file:
            configuration = toml.load(file)

        set_logging(configuration['Global']['log_file'],configuration['Global']['log_level'])
        tod_processing = create_tod_processing(configuration) 
        tod_processing.run_tod()
    
    if False:
        astro_processing = create_astro_cal(targets='processed_runlists/level2_galactic.txt',
                                        source='CasA',figure_directory='figures_CasA',
                                        cal_files = 'processed_runlists/level2_CasA.txt')
        astro_processing.run_astro_cal() 

if __name__ == "__main__": 
    main()
