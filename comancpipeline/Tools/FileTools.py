import numpy as np
from matplotlib import pyplot
import subprocess
from datetime import datetime

def query_sql(server, script_location,suffix='_Level2Cont'):
    """
    query the sql database at OVRO
    """
    sql = f'python {script_location}' 
    command = ['ssh -v',server, '"{}"'.format(sql)]
    p = subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE)
    pipe, err = p.communicate()

    obsinfo = {}
    for line in pipe.splitlines():
        if len(line.split()) != 4:
            continue
        obsid, target,day,time = [l.decode() for l in line.split()]
    
        date = datetime.strptime(f'{day} {time}','%Y-%m-%d %H:%M:%S.%f')
        dt = date.strftime('%Y-%m-%d-%H%M%S')
        obsid = int(obsid)
        filename = f'comap-{obsid:07d}-{dt}{suffix}.hd5'
        obsinfo[filename] = target

    return obsinfo

import h5py
class h5py_visitor_func_class:

    def __init__(self):
        self.data = {}

    def __call__(self, name, node):
        if isinstance(node, h5py.Dataset) and not name in self.data:
            self.data[names] = node[...]
        
