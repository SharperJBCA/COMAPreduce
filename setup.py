from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension
import os

try:
    slalib_path = os.environ['SLALIB_LIBS']
except KeyError:
    slalib_path = '/star/lib' # default path of Manchester machines
    print('Warning: No SLALIB_LIBS environment variable set, assuming: {}'.format(slalib_path))

pysla = Extension(name = 'comancpipeline.Tools.pysla', 
                  sources = ['comancpipeline/Tools/pysla.f90'],
                  libraries=['sla'],
                  library_dirs =['{}'.format(slalib_path)],
                  f2py_options = [],
                  extra_f90_compile_args=['-L{}'.format(slalib_path),'-lsla'])
config = {'name':'comancpipeline',
          'version':'0.1dev',
          'packages':['comancpipeline','comancpipeline.Analysis','comancpipeline.Tools'],
          'ext_modules':[pysla]}



setup(**config)

#    name='comancpipeline',
#    version='0.1dev',
#    packages=['comancpipeline.Analysis','comancpipeline.Tools'])

