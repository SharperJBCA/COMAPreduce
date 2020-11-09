# Tools for logging errors, outputs, etc...
import time
from astropy.time import Time
import traceback
class Logger:

    def __init__(self,logfile):
        
        self.log = open(logfile,'w')


    def __call__(self,filename, error):
        
        tb = traceback.extract_tb(error.__traceback__)
        error_s = '\n'.join(traceback.format_list(tb))
        tnow = Time(time.time(),format='unix').isot
        self.log.write('{} : \t Error processing: {}\n'.format(tnow,filename))
        self.log.write('{} : \t class:{} msg:{} traceback:\n{}\n'.format(tnow, error.__class__, error, error_s  ))
        self.log.write('##########\n')
        

    def write(self,output):

        tnow = Time(time.time(),format='unix').isot

        self.log.write('{} : \t {}\n'.format(tnow,output))

    def __del__(self):
        self.log.close()
