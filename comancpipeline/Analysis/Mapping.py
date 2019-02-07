from comancpipeline.Analysis.BaseClasses import DataStructure
import h5py

class SimpleMap(object):

    def __init__(self, xpix=1, ypix=1, cdeltx=None, cdelty=None):
        self.xpix = xpix
        self.ypix = ypix
        self.cdeltx = cdeltx
        self.cdelty = cdelty

    def __call__(self, container, filename=''):
        self.run(container)
        self.filename = filename

    def run(self, container):
        container['pointing/azActual']
