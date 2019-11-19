import numpy as np
import re
import glob
import datetime
import pandas as pd

class Parser(object):
    """
    Raw parser class for reading .ini files
    """

    def __init__(self, filename, delims=[':','=']):

        # Initialise Parser as an object, needed?
        super(Parser, self)
        
        # Parameters stores all the parameter keys and values
        self.parameters = {}

        # Set the k <delim> value format in ini file (default: ':' or '=')
        self.delims = delims

        # Open the file object
        self.file = open(filename, 'r')
        self.ReadLines() # Read the file
        self.file.close()

    def __str__(self):
        """
        Print all key names
        """
        headers = [k+',\n' for k in list(self.parameters.keys())]
        return '{'+''.join(headers)+'}'

    def __setitem__(self,k,v):
        self.parameters[k] = v
     
    def __getitem__(self,k):
        try:
            return self.parameters[k]
        except KeyError:
            raise AttributeError ('Unknown key: {}'.format(k))
        
    def __contains__(self, k):
        return k in self.parameters
 
    def items(self):
        return self.parameters.items()
        
    def ReadLines(self):
        """
        Reads the ini file line by line.
        """
        
        for line in self.file:
            #First remove comments
            line_nocomments = line.split('#')[0].strip() # Remove all white-space

            if len(line_nocomments) > 0: # Catch lines with all comments.
                if (line_nocomments[0] == '[') & (line_nocomments[-1] == ']'):
                    #Now check for Headers
                    thisHeader = re.split('\\[|\\]', line_nocomments)[1]
                    if thisHeader not in self.parameters:
                        self.parameters[thisHeader] = {}
                else:
                    #Now fill the headers
                    for _delim in self.delims:
                        try:
                            splits = line_nocomments.split(_delim)
                            if len(splits) == 2:
                                keyword, value = line_nocomments.split(_delim)
                                delim = _delim
                            elif len(splits) > 2:
                                keyword = splits[0]
                                value = '{}'.format(_delim).join(splits[1:len(splits)])
                                delim = _delim
                            else:
                                continue

                             # no point in doing other delimiters
                        except ValueError:
                            print('Failed', keyword,value)
                            continue
                    
                    if isinstance(value,list):
                        value = '{}'.format(delim).join(value)

                    #Want to check for arrays of values split by commas
                    value = value.replace(' ', '').split(',')
                    keyword = keyword.replace(' ', '')

                    if len(value) > 1:
                        self.parameters[thisHeader][keyword] = []
                        
                        for v in value:
                            if v == 'None':
                                self.parameters[thisHeader][keyword] += [None]
                            elif ('True' in v):
                                self.parameters[thisHeader][keyword] += [True]            
                            elif ('False' in v):
                                self.parameters[thisHeader][keyword] += [False]         
                            else:
                                try:
                                    self.parameters[thisHeader][keyword] += [float(v)]
                                except ValueError:
                                    self.parameters[thisHeader][keyword] += [v]
                    else:
                        for v in value:
                            if v == 'None':
                                self.parameters[thisHeader][keyword] = None
                            elif ('True' in value):
                                self.parameters[thisHeader][keyword] = True          
                            elif ('False' in value):
                                self.parameters[thisHeader][keyword] = False       
                            else:
                                try:
                                    self.parameters[thisHeader][keyword] = float(v)
                                except ValueError:
                                    self.parameters[thisHeader][keyword] = v
                        
                        

        self.file.close()
