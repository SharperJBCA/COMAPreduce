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
                  #libraries=['sla'],
                  #library_dirs =['{}'.format(slalib_path)],
                  f2py_options = [])
                 # extra_link_args=['-Wl,-rpath,{}'.format(slalib_path)])
#'Iglibc_fix.h',
                                      
ffuncs = Extension(name = 'comancpipeline.Tools.ffuncs', 
                  sources = ['comancpipeline/Tools/ffuncs.f90'])

# alglib = Extension('comancpipeline.Tools.alglib_optimize',
#                    ['comancpipeline/Tools/alglib_optimize.pyx'],
#                    library_dirs=["/local/scratch/sharper/etc/lib"],
#                    include_dirs=["/local/scratch/sharper/etc/lib"],
#                    language="c++",
#                    extra_compile_args=['-fopenmp'],
#                    extra_link_args=['-fopenmp']
#                )
#'-lAlglib',

filters = Extension(name = 'comancpipeline.Tools.median_filter.medfilt', 
                    include_dirs=[np.get_include()],
                    sources = ['comancpipeline/Tools/median_filter/medfilt.pyx'])

binFuncs = Extension(name='comancpipeline.Tools.binFuncs',
                     include_dirs=[np.get_include()],
                     sources=['comancpipeline/Tools/binFuncs.pyx'])


config = {'name':'comancpipeline',
          'version':__version__,
          'packages':['comancpipeline',
                      'comancpipeline.Analysis',
                      'comancpipeline.data',
                      'comancpipeline.Tools',
                      'comancpipeline.MapMaking'],
          'package_data':{'':["*.dat"]},
          'include_package_data':True,
          'ext_modules':cythonize([ffuncs,pysla, filters,binFuncs],
                                  compiler_directives={'language_level':"3"})}



setup(**config)
print(__version__)
