#!/usr/bin/env python3
# ===============================================================================
# SciptName     : process.py
# ScriptDir     : /src/core/process.py
# Author(s)     : T.J.Rennie
# Description   : Base class for callable processes inside the pipeline
# ===============================================================================

# P-I: IMPORTS
import textwrap
import numpy as np
from mapext.core.mapClass import astroMap

# I: PROCESS CLASS

class Process:

    def __init__(self,*args,
                 RUNDICT=None,
                 cpId=None,
                 **kwargs):
        '''
        Initialises class with process parameters and general run parameters.

        INPUTS:
        genParams   : ARG - Dictionary of general pipeline parameters.
                      e.g. run name, working directory, ...)
        proParams   : ARG - Process specific parameters.
                      e.g. iterations, step length, amplitude
        '''
        # DEFINE RUN VARIABLES
        self.RUNDICT = RUNDICT
        self.__cpId__ = cpId

        # RUN THE FUNCTION
        self.run(*args, **kwargs)

    def run(self,*args,testText='testingTesting123',failRun=False,rtnInfo=True,
            **kwargs):
        '''
        Process to be run.

        INPUTS:
        KWARGS      : Keyword arguments should represent process parameters that
                      must be pre-determined and may be customised through
                      proParams and ovrParams for global and individual settings
                      respectively
        OUTPUTS:
        data product
        '''
        if failRun == False:
            print(testText)

    def ver_print(self,text,ver_no=0):
        '''
        VERPRINT NUMBER CONVENTION:
        0  | OTHER - not in any of the below categories (huh???)
        1  | PRIMARY OUTPUT - essential output code to terminal (nice)
        2  | DEBUG INFORMATION - info for debugging issues (hmmm)
        3  | ERROR - program runs but output may be limited (most irregular)
        4  | TERMINAL ERROR - stops program running (Whoops?)
        '''
        if self.__cpId__ != False:
            startline = 'C{:02}P{:03};{:->20};'.format(self.__cpId__[0],
                        self.__cpId__[1], self.__name__)  # 30 spaces
        else:
            startline = '{:->30};'.format(self.__cpId__[0],
                        self.__cpId__[1], self.__name__)

        if ver_no == 3:
            text = 'ERROR; ' + text
        elif ver_no == 4:
            text = 'TERMINAL ERROR; ' + text

        if len(text) > 50:
            newline = '\n' + ' ' * 30
            text = textwrap.fill(text=text, width=50)
            text = text.replace('\n', newline)

        if type(self.genParams['verbose']) is bool:
            if self.genParams['verbose']:
                print(startline, text)
        else:
            if ver_no in self.genParams['verbose']:
                print(startline, text)
