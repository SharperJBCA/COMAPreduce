from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension
import os
import numpy as np
from Cython.Build import cythonize
try:
    slalib_path = os.environ['SLALIB_LIBS']
except KeyError:
    slalib_path = '/star/lib' # default path of Manchester machines
    print('Warning: No SLALIB_LIBS environment variable set, assuming: {}'.format(slalib_path))

filters = Extension(name = 'comancpipeline.Tools.filters', 
                    include_dirs=[np.get_include()],
                    sources = ['comancpipeline/Tools/filters.pyx'])

pysla = Extension(name = 'comancpipeline.Tools.pysla', 
                  sources = ['comancpipeline/Tools/pysla.f90'],
                  libraries=['sla'],
                  library_dirs =['{}'.format(slalib_path)],
                  f2py_options = [],
                  extra_f90_compile_args=['-L{}'.format(slalib_path)])
#,'-L{}/libsla.so'.format(slalib_path)])

ffuncs = Extension(name = 'comancpipeline.Tools.ffuncs', 
                  sources = ['comancpipeline/Tools/ffuncs.f90'])

alglib = Extension('comancpipeline.Tools.alglib_optimize',
                   ['comancpipeline/Tools/alglib_optimize.pyx'],
                   library_dirs=["/local/scratch/sharper/etc/lib"],
                   include_dirs=["."],
                   language="c++",
                   extra_compile_args=['-lAlglib','-fopenmp'],
                   extra_link_args=['-lAlglib','-fopenmp']
               )


filters = Extension(name = 'comancpipeline.Tools.median_filter.medfilt', 
                    include_dirs=[np.get_include()],
                    sources = ['comancpipeline/Tools/median_filter/medfilt.pyx'])

config = {'name':'comancpipeline',
          'version':'0.1dev',
          'packages':['comancpipeline','comancpipeline.Analysis','comancpipeline.Tools'],
          'ext_modules':cythonize([ffuncs,pysla, filters,alglib])}



setup(**config)

#    name='comancpipeline',
#    version='0.1dev',
#    packages=['comancpipeline.Analysis','comancpipeline.Tools'])

 
