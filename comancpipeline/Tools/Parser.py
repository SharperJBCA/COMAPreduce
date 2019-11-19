# Scripts for parsing the parameter file.
import numpy as np
from comancpipeline.Tools import ParserClass
from  comancpipeline import Analysis 

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
    """
    Returns uninitalise class instance with modulename.classname format
    """
    
    modulename, classname = strname.split('.')
    module_ = getattr(Analysis,modulename)
    class_  = getattr(module_,classname)
    return class_
        

def parse_parameters(filename):
    """
    Take a parameter file, return list of initalised objects (jobs) and input parameters 
    """

    # read in the parameters
    mainInput = ParserClass.Parser(filename)

    # Generate a filelist to loop over
    filelist = np.loadtxt(mainInput['Inputs']['filelist'],dtype=str,ndmin=1)
    if isinstance(mainInput['Inputs']['data_dir'], type(None)):
        filelist = [filename for filename in filelist]
    else:
        filelist = ['{}/{}'.format(mainInput['Inputs']['data_dir'],
                                   filename.split('/')[-1]) for filename in filelist]
                               
    # Some items should always be a list
    if not isinstance(mainInput['Inputs']['pipeline'], list):
        mainInput['Inputs']['pipeline'] = [mainInput['Inputs']['pipeline']]
    # Get the class names (modulename, classname)
    jobnames = [c for c in mainInput['Inputs']['pipeline']]

    print(mainInput['Inputs']['pipeline'])

    # Read the class parameter file
    classInput = ParserClass.Parser(mainInput['Inputs']['classParameters'])

    # Initalise the classes : classInput are the kwargs to initiate classes
    jobs = []
    for job in jobnames:
        jobs += [getClass(job)(**classInput[job])]

    return jobs, filelist, mainInput, classInput

def parse_split(config, field):

    print(config,field)
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
    else:
        return None, 0
    
