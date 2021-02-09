import numpy as np
import re

class Parser(object):
    """
    """

    def __init__(self, filename):
        super(Parser, self)
        
        self.infodict = {}

        self.file = open(filename, 'r')
        self.ReadLines()

    def __str__(self):
        headers = [k+',\n' for k in self.infodict.keys()]
        return '{'+''.join(headers)+'}'

    def __setitem__(self,k,v):
        self.infodict[k] = v
    
    def __getitem__(self,k):
        try:
            return self.infodict[k]
        except KeyError:
            raise AttributeError ('Unknown key: {}'.format(k))
        
    def __contains__(self, k):
        return k in self.infodict
 
    def items(self):
        return self.infodict.items()
        
    def ReadLines(self):
        """
        """
        
        
        for line in self.file:
            #First remove comments
            line_nocomments = line.split('#')[0].strip()
            if len(line_nocomments) > 0:

                if (line_nocomments[0] == '[') & (line_nocomments[-1] == ']'):
                    #Now check for Headers
                    thisHeader = re.split('\\[|\\]', line_nocomments)[1]
                    if thisHeader not in self.infodict:
                        self.infodict[thisHeader] = {}
                else:
                    #Now fill the headers
                    delims = [':','=']
                    for _delim in delims:
                        try:
                            index = line_nocomments.index(_delim)
                            keyword = line_nocomments[:index].strip()
                            value = line_nocomments[index+1:].strip()
                            #keyword, value = line_nocomments.split(_delim)
                            delim = _delim
                            break # no point in doing other delimiters
                        except ValueError:
                            continue
                    
                    #Want to check for arrays of values split by commas
                    value = value.replace(' ', '').split(',')
                    keyword = keyword.replace(' ', '')


                    if len(value) > 1:
                        self.infodict[thisHeader][keyword] = []
                        
                        for v in value:
                            if v == '':
                                continue
                            elif v == 'None':
                                self.infodict[thisHeader][keyword] += [None]
                            elif (v.strip() == 'True'):
                                self.infodict[thisHeader][keyword] += [True]            
                            elif (v.strip() == 'False'):
                                self.infodict[thisHeader][keyword] += [False]         
                            else:
                                try:
                                    self.infodict[thisHeader][keyword] += [float(v)]
                                except ValueError:
                                    self.infodict[thisHeader][keyword] += [v]
                    else:
                        for v in value:
                            if v == 'None':
                                self.infodict[thisHeader][keyword] = None
                            elif (v.strip() == 'True'):
                                self.infodict[thisHeader][keyword] = True          
                            elif (v.strip() == 'False'):
                                self.infodict[thisHeader][keyword] = False       
                            else:
                                try:
                                    self.infodict[thisHeader][keyword] = float(v)
                                except ValueError:
                                    self.infodict[thisHeader][keyword] = v
                        
                        
        self.file.close()
