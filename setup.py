from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension

pysla = Extension(name = 'comancpipeline.Tools.pysla', 
                  sources = ['comancpipeline/Tools/pysla.f90'],
                  libraries=['sla'],
                  library_dirs =['/star/lib'],
                  f2py_options = ['--f90exec=gfortran'],
                  extra_f90_compile_args=['-L/star/lib','-lsla'])
config = {'name':'comancpipeline',
          'version':'0.1dev',
          'packages':['comancpipeline.Analysis','comancpipeline.Tools'],
          'ext_modules':[pysla]}



setup(**config)

#    name='comancpipeline',
#    version='0.1dev',
#    packages=['comancpipeline.Analysis','comancpipeline.Tools'])

