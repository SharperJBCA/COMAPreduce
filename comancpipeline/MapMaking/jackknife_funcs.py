import numpy as np
import os
from datetime import datetime
from astropy.time import Time
import glob

def get_images(map_directory,modes=None):
    file_fmt = '{}/{}/{}_Feeds{}_Band{}{}.fits'

    if isinstance(modes,type(None)):
        modes = np.unique([os.path.basename(f).split('_')[0] for f in glob.glob(map_directory+'/*')])
    #modes = np.unique(mode_list)
    #modes = ['All','FirstHalf','SecondHalf']

    filenames = {}
    for mode in modes:
        filenames[mode] = []
        for feed in range(1,20):
            for iband in range(8):
                filenames[mode] += [file_fmt.format(map_directory, 
                                                    mode+'_Feed{:02d}'.format(feed),
                                                    mode+'_Feed{:02d}'.format(feed),
                                                    feed,
                                                    iband,
                                                    '_MFS1500step')]
    return filenames

def create_parameter_file(filelist='', name='', feeds='', 
                          map_directory='', offset_size='',
                          nypix='',nxpix='',cdelt='',crpix='',
                          crval='',ctype=''):
    
    parameter_file = """[Inputs]
    filelist :  {filelist} # Filelist of level 2 continuum datasets
    title : {name} # Prefix for the output maps

    feeds :        {feeds}
    feed_weights : 1,1,1,1,1,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1

    saveoffsets : False # Save the fitted offsets to disk (Not yet implemented)
    offsets_directory : None # Directory for saved offsets

    maps_directory : {map_directory} # Output directory
    calibration_source : taua

    [Destriper]
    
    offset : {offset_size}
    niter : 1200
    verbose : True
    threshold : -5
    
    nypix : {nypix}
    nxpix : {nxpix}
    cdelt : {cdelt}
    crpix : {crpix}
    crval : {crval}
    ctype : {ctype}

    [ReadData]
    
    ifeature : 5
    iband : 7
    keeptod : True
    subtract_sky : False
    flag_spikes : True
    """.format(filelist=filelist, name=name, feeds=feeds, 
               map_directory=map_directory, offset_size=offset_size,
               nypix=nypix,nxpix=nxpix,cdelt=cdelt,crpix=crpix,
               crval=crval,ctype=ctype)

    return parameter_file

def create_parameter_files(source_name,filelist_info, map_info):

    pfile_names = {}
    for upper_key, upper_group in filelist_info.items():
        for k,v in upper_group.items():
            feed = int(k[-2:])
            output_dir = 'jackknives/{}/{}/'.format(source_name,k)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filelist_name = '{}/{}_filelist.txt'.format(output_dir,k)
            filelist = open(filelist_name,'w')
            for line in v:
                filelist.write('{}\n'.format(line))
            filelist.close()
            parameters = create_parameter_file(filelist=filelist_name,
                                               name=k,
                                               feeds=feed,
                                               map_directory=output_dir,
                                               offset_size=50,
                                               **map_info)
                         
            pfile_name = '{}/{}_Parameters.ini'.format(output_dir,k)
            pfile = open(pfile_name,'w')
            pfile.write(parameters)
            pfile.close()
            pfile_names[k] = pfile_name
    return pfile_names

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
    python run_destriper.py {script}
    """.format(name=name, email=email, work_dir=work_dir, script=script)

    script_name = 'pbs_dir/{}_pbs.script'.format(name)
    fout = open(script_name,'w')
    fout.write(pbs_script)
    fout.close()
    return script_name

def split_all(filelist,nfeeds=19):

    return {'All':{'All_Feed{:02d}'.format(i+1):filelist for i in range(nfeeds)}}

def split_quartiles(filelist,nfeeds = 19):

    output = []
    names = ['FirstQuarter','SecondQuarter','ThirdQuarter','FourthQuarter']
    edges = [int(0), len(filelist)//4,len(filelist)//4*2,len(filelist)//4*3, len(filelist)]

    output = {k:{} for k in names}
    for (name,start,end) in zip(names,edges[:-1],edges[1:]):
        for i in range(nfeeds):
            output[name][('{}_Feed{:02d}'.format(name,i+1))] = filelist[start:end]
    return output


def split_half(filelist,nfeeds = 19):

    output = []
    names = ['FirstHalf','SecondHalf']
    edges = [int(0), len(filelist)//2, len(filelist)]

    output = {k:{} for k in names}
    for (name,start,end) in zip(names,edges[:-1],edges[1:]):
        for i in range(nfeeds):
            output[name][('{}_Feed{:02d}'.format(name,i+1))] = filelist[start:end]
    return output


def split_months(filelist,nfeeds = 19):

    dates = []
    keys = []
    for filename in filelist:
        basename = os.path.basename(filename)
        _,obsid,year,month,day,hhmmss = basename.split('-')
        hhmmss = hhmmss.split('_')[0]
        date = datetime(int(year),int(month),int(day),
                        int(hhmmss[:2]),int(hhmmss[2:4]),int(hhmmss[4:]))
        dates += [date]
        keys  += [str(date.year)+str(date.month)]
    unique_keys = np.unique(keys)

    keys = np.array(keys)
    indices = []
    for key in unique_keys:
        indices += [np.where(keys == key)[0]]

    output = {k:{} for k in unique_keys}
    for (name,select) in zip(unique_keys,indices):
        for i in range(nfeeds):
            output[name][('{}_Feed{:02d}'.format(name,i+1))] = filelist[select]
    return output


def split_filelists(filename):

    filelist = np.loadtxt(filename,dtype=str)

    modes = [split_all,split_half,split_months]#,split_quartiles]
    output = {}
    for mode in modes:
        output = {**output,**mode(filelist)}
    return output
