# Tools for logging errors, outputs, etc...
import time
from astropy.time import Time
import traceback
import os

class Logger:

    def __init__(self,logfile):
        
        errfile = logfile.split('/')[-1]
        logdir = logfile.split(errfile)[0]

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.log = open(logfile,'w')
        self.err_log = open(logdir+'errors_'+errfile,'w')

    def __call__(self,__str__,error=None):

        if not isinstance(error,type(None)):
            self.error_call(__str__,error)
        else:
            self.str_call(__str__)
        self.log.flush()
        os.fsync(self.log.fileno())

    def error_call(self,filename, error):
        
        tb = traceback.extract_tb(error.__traceback__)
        error_s = '\n'.join(traceback.format_list(tb))
        tnow = Time(time.time(),format='unix').isot
        self.err_log.write('{} : \t Error processing: {}\n'.format(tnow,filename))
        self.err_log.write('{} : \t class:{} msg:{} traceback:\n{}\n'.format(tnow, error.__class__, error, error_s  ))
        self.err_log.flush()
        os.fsync(self.err_log.fileno())
        
    def str_call(self, __str__):
        tnow = Time(time.time(),format='unix').isot
        self.log.write('{} : \t {}\n'.format(tnow,__str__))


    def write(self,output):

        tnow = Time(time.time(),format='unix').isot

        self.log.write('{} : \t {}\n'.format(tnow,output))

    def __del__(self):
        self.log.close()
