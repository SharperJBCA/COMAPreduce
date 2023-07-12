from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension
import os
import numpy as np
from Cython.Build import cythonize

# Capture the current git commit to use as version
exec(open("comancpipeline/version.py").read())

try:
    slalib_path = os.environ['SLALIB_LIBS']
except KeyError:
    slalib_path = '/star/lib' # default path of Manchester machines
    print('Warning: No SLALIB_LIBS environment variable set, assuming: {}'.format(slalib_path))

filters = Extension(name = 'comancpipeline.Tools.filters', 
                    include_dirs=[np.get_include()],
                    sources = ['comancpipeline/Tools/filters.pyx'])

pysla = Extension(name = 'comancpipeline.Tools.pysla', 
                  sources = ['comancpipeline/Tools/pysla.f90','comancpipeline/Tools/sla.f'],
                  f2py_options = [])
                                      
ffuncs = Extension(name = 'comancpipeline.Tools.ffuncs',
                  sources = ['comancpipeline/Tools/ffuncs.f90'])

filters = Extension(name = 'comancpipeline.Tools.median_filter.medfilt', 
                    include_dirs=[np.get_include()],
                    sources = ['comancpipeline/Tools/median_filter/medfilt.pyx'])

binFuncs = Extension(name='comancpipeline.Tools.binFuncs',
                     include_dirs=[np.get_include()],
                     sources=['comancpipeline/Tools/binFuncs.pyx'])


config = {'name':'comancpipeline',
          'version':__version__,
          'packages':['comancpipeline',
                      'comancpipeline.RRLs',
                      'comancpipeline.Analysis',
                      'comancpipeline.Summary',
                      'comancpipeline.data',
                      'comancpipeline.Tools',
                      'comancpipeline.Simulations',
                      'comancpipeline.SEDs',
                      'comancpipeline.SEDs.amemodels',
                      'comancpipeline.MapMaking'],
          'package_data':{'':["*.dat","gains.hd5"]},
          'include_package_data':True,
          'ext_modules':cythonize([filters,binFuncs],
                                  compiler_directives={'language_level':"3"})}



setup(**config)
print(__version__)
