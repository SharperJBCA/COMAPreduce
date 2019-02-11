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

    targets = Parameters.get('Inputs', 'targets').split(',')
    targets = [t.lower() for t in targets]

    selectors = []
    for key in order:
        if Parameters.has_section(key):
            valDict = dict(Parameters.items(key))
            checkTypes(valDict)

            c = getClass(key)
            
            selectors += [c(**valDict)]
        else:
            c = getClass(key)
            selectors += [c()]

    return selectors, targets, Parameters

def parse_split(config, field):

    if config.has_section(field):
        if config.has_option(field, 'selectAxes'):
            selectAxes = config.get(field, 'selectAxes')
            if selectAxes.lower() == 'none':
                selectAxes = None
            else:
                selectAxes = selectAxes.split(',')
                selectAxes = [int(s) for s in selectAxes]
        else:
            selectAxes = None
        if config.has_option(field, 'splitAxis'):
            splitAxis = config.get(field, 'splitAxis')
            if splitAxis.lower() == 'none':
                splitAxis = None
            else:
                splitAxis = int(splitAxis)
        else:
            splitAxis = 0

    return selectAxes, splitAxis
    
