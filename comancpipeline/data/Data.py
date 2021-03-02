import numpy as np
import comancpipeline
import pathlib
path = pathlib.Path(comancpipeline.__file__).resolve().parent
average_beam_widths = np.loadtxt(f'{path}/data/AverageBeamWidths.dat',skiprows=1)

average_beam_widths = {int(d[0]):d[1:] for d in average_beam_widths}


feed_positions = np.loadtxt(f'{path}/data/COMAP_FEEDS.dat')

feed_positions = {int(d[0]):d[1:] for d in feed_positions}
