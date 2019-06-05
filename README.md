# Manchester COMAP Data Reduction Pipeline

Routines and procedures for calibrating and reducing raw COMAP spectroscopic single-dish radio data.

### Prerequisites

Prerequisites are listed in an approximate order in which they should be installed: 
* [Python 3.X.X](https://www.python.org/downloads/) 
* [SLALIB](http://starlink.eao.hawaii.edu/starlink/2018ADownload) - Astronomical libraries for any spherical trig.
* [OPENMPI](https://www.open-mpi.org)/[MPICH](https://www.mpich.org) - The pipeline requires MPI to run, either backend will work fine.
* [MPI4PY](mpi4py.readthedocs.io) - MPI4Py must be compiled against the mpi library above.
* [HDF5 --parallel-enabled](http://docs.h5py.org/en/stable/mpi.html) - An mpi ready version of the HDF5 libraries (see link for instructions)
* [H5PY](http://www.h5py.org) - Must be compiled against above HDF5 library (not the one included with [anaconda](https://www.anaconda.com/distribution/) )
* [HEALPY](healpy.readthedocs.io) - Either against your local version of [HEALPix](https://healpix.jpl.nasa.gov) or a pre-compiled version from pip.

Other standard libraries such as [NumPy](https://www.numpy.org), [SciPy](https://www.scipy.org), [matplotlib](https://matplotlib.org), etc... are assumed to be installed already.

### Installing

After all prerequistites are installed first install the library by
```
cd /path/to/COMAPreduce/
python setup.py install
```
You will then need to setup a working directory that will contain:
* run.py
* All the .ini files
* COMAP_FEEDS.dat
```
mkdir /path/to/working/directory/
cp /path/to/COMAPreduce/run.py  /path/to/working/directory/
cp /path/to/COMAPreduce/*.ini /path/to/working/directory/
cp /path/to/COMAPreduce/COMAP_FEEDS.dat /path/to/working/directory/
```
### Running

Three example parameter files (the .ini files) have been provided.

To run the pipeline you will need to both choose a parameter file and generate a list of files. In 
```
COMAPreduce/comancpipeline/scripts/io/
```
there is script called createFileList.py that can help to generate a filelist, it is executed as
```
python createFileList -D /path/to/level1/files -o "string describing observation you need (e.g. TauA)" -F output_filelist_name.dat
```

Finally, to run the pipeline you invoke in your working directory
```
python run.py -P parameterfile.ini -F filelist.dat
```
and for MPI executions
```
mpiexec -n X python run.py -P parameterfile.ini -F filelist.dat
```
where X is the number of cores you want to use.
