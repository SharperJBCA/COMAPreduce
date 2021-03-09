import numpy as np
import comancpipeline
import pathlib
import h5py
path = pathlib.Path(comancpipeline.__file__).resolve().parent
average_beam_widths = np.loadtxt(f'{path}/data/AverageBeamWidths.dat',skiprows=1)

average_beam_widths = {int(d[0]):d[1:] for d in average_beam_widths}


feed_positions = np.loadtxt(f'{path}/data/COMAP_FEEDS.dat')

feed_positions = {int(d[0]):d[1:] for d in feed_positions}


feed_gains_hd5 = h5py.File(f'{path}/data/gains.hd5','r')

feed_gains = {k:{'obsids':v['obsids'][...],'gains':v['gains'][...]} for k,v in feed_gains_hd5.items()}

feed_gains_hd5.close()
