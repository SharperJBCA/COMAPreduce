# Scripts for parsing the parameter file.

import configparser
import comancpipeline.Analysis as Analysis

def checkTypes(valDict):
    for k, v in valDict.items():
        # is a bool?
        if v.lower() == 'true':
            valDict[k] = True
        elif v.lower() == 'false':
            valDict[k] = False
            
        # is it an int?
        try:
            valDict[k] = int(v)
        except ValueError:
            pass
        # is it a float?
        try:
            valDict[k] = float(v)
        except ValueError:
            pass
        #... then it is a str

def getClass(strname):
    
    modulename, classname = strname.split('.')
    module_ = getattr(Analysis,modulename)
    class_  = getattr(module_,classname)
    return class_

def parse_parameters(filename):

    Parameters = configparser.ConfigParser()
    Parameters.read(filename)

    order = Parameters.get('Inputs', 'order').split(',')

    selectors = []
    for key in order:
        valDict = dict(Parameters.items(key))
        checkTypes(valDict)

        c = getClass(key)

        selectors += [c(**valDict)]

    return selectors
