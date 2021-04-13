import click
import SourceFitting
import h5py

def main_cli():
    pass

def fit_source(filename=None):
    """
    """

    filename = '/scratch/nas_comap3/sharper/COMAP/data/comap-0018282-2021-03-06-172503.hd5'
    fs_obj = SourceFitting.FitSource(output_dir='data_out',logger=print,feeds=[1])

    hdf5_obj = h5py.File(filename,'r')
    fs_obj(hdf5_obj)
    hdf5_obj.close()

if __name__ == "__main__":
    fit_source()
