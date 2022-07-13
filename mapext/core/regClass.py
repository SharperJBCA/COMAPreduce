#!/usr/bin/env python3
# ===============================================================================
# SciptName     : srcClass.py
# ScriptDir     : /src/core/srcClass.py
# Author(s)     : T.J.Rennie
# Description   : astroSrc class for holding astronomical source data and
#                 permitting easy boundsing and parameter retrieval
# ===============================================================================

# P-I: IMPORTS
from astropy.coordinates import SkyCoord
import h5py
import inspect
import numpy as np
from pydoc import locate

from mapext.core.usefulFunctions import h5PullDict

# I: PROCESS CLASS

class astroReg():
    '''
    Class similar to astroSrc, but to hold more diffuse sources that require bounds to be set that are not circular/gaussian
    '''

    def __init__(self,
                 coord=None,name=None,referance=None,
                 fluxes=None,component_fluxes=None,
                 reg_bounds=None,
                 notes=None,
                 test=False):

        if fluxes != None:
            self.flux =  fluxes
        else:
            self.flux = np.zeros((0)).astype([('method','S50'),('survey','S50'),('name','S50'),('nu', float), ('Sv', float), ('Sv_e', float)])

        if component_fluxes != None:
            self.comp_flux =  component_fluxes
        else:
            self.comp_flux = np.zeros((0)).astype([('component','S50'),('name','S50'),('nu', float), ('Sv', float), ('Sv_e', float)])

        if reg_bounds != None:
            self.reg_bounds =  np.array(reg_bounds, dtype=([('survey','S50'),('name','S50'),('nu', float),('bounds', float, (101,2)),('bkg', float, (3)),('bkg2', float, (101,2))]))
        else:
            self.reg_bounds = np.zeros((0)).astype([('survey','S50'),('name','S50'),('nu', float),('bounds', float, (101,2)),('bkg', float, (3)),('bkg2', float, (101,2))])

        if coord != None:
            self.COORD = coord
        else:
            cen = np.nanmean(self.reg_bounds[0]['bounds'],axis=0)
            self.COORD = SkyCoord(*cen,frame='galactic',unit='degree')

        if name != None:
            self.NAME = name
        else:
            if self.COORD.b.degree<0:
                sign = '+'
            else:
                sign = '-'
            self.NAME = 'G{:06.2f}{}{:05.2f}'.format(self.COORD.l.degree,sign,abs(self.COORD.b.degree))
        self.REFS = referance

        self.NOTE = notes

    def load_src(self,srcfile=None):
        if srcfile == None:
            srcfile = 'SRCHDF5/'+self.ID
        f = h5py.File(srcfile,'r')
        params = h5PullDict(f)
        f.close()
        self.COORD = f['general/COORD']
        self.NAME = f['general/NAME']
        self.add_flux(f['fluxes/total_flux'])

    def update(self,
               coord=None,name=None,referance=None,
               fluxes=None,component_fluxes=None,
               reg_bounds=None,
               notes=None,
               test=False):
        self.COORD = coord
        if name != None:
            self.NAME = name
        if reference != None:
            self.REFS = referance
        if fluxes != None:
            self.add_flux(fluxes)
        if component_fluxes != None:
            self.comp_flux =  component_fluxes
        if reg_bounds != None:
            self.reg_bounds =  reg_bounds
        if notes != None:
            self.NOTE = notes

    def add_reg_bounds(self, data):
        data = np.array(data,dtype=[('survey','S50'),('name','S50'),('nu', float), ('bounds', float, (101,2)),('bkg', float, (3)),('bkg2', float, (101,2))])
        self.reg_bounds = np.concatenate((self.reg_bounds,data))

    def add_flux(self, data):
        data = np.array(data,dtype=[('method','S50'),('survey','S50'),('name','S50'),('nu', float), ('Sv', float), ('Sv_e', float)])
        self.flux = np.concatenate((self.flux,data))

    def add_compflux(self, data):
        data = np.array(data,dtype=[('component','S50'),('name','S50'),('nu', float), ('Sv', float), ('Sv_e', float)])
        self.flux = np.concatenate((self.flux,data))

    def return_dictionary(self):
        out_dict =  {'general':    {'NAME':         self.NAME,
                                    'COORD':        [self.COORD.l.degree,
                                                     self.COORD.b.degree],
                                    'REFS':         self.REFS},
                     'fluxes':     {'total_flux':   self.flux,
                                    'comp_flux':    self.comp_flux},
                     'bounds':     {'reg_bounds':    self.reg_bounds},
                     'note':        self.NOTE}
        return out_dict

    # def change_bounds(self,boundsname):
    #     if boundsname in list(self.bounds_alt.keys()):
    #         self.def_bounds(self.bounds_alt[boundsname])
    #
    # def def_bounds(self, bounds_dict):
    #     '''
    #     Function to set the bounds used to define the emission of the source in
    #     frequency space. Set in two parts: emission bounds and emission parameter
    #
    #     INPUTS:
    #     bounds_dict  : Dictionary of the form
    #            {EMISSION_TYPE:{n:{PARAMTER_NAME : PARAMETER_VALUE,...},...},...}
    #     '''
    #     # reset bounds
    #     self.emissionbounds = 0
    #     self.emission_comp = {}
    #
    #     # # define bounds
    #     # from src.core.emissionMech import bounds_available as allowed_bounds
    #
    #     def get_function_paramaters(func):
    #         from inspect import getargspec
    #         params = getargspec(func).args
    #         nparams = len(params)
    #         return params, nparams
    #
    #     for emission_name in bounds_dict:
    #         # if emission_name not in allowed_bounds:
    #         #     print('WARNING: emission bounds {} not recognised from emission file'.format(
    #         #         emission_name))
    #         #     return
    #         self.emission_comp[emission_name] = {}
    #         self.emission_comp[emission_name]['nmod'] = np.max(
    #             len(list(bounds_dict[emission_name])))
    #         self.emission_comp[emission_name]['function'] = locate(
    #             'src.core.emissionMech.{}'.format(emission_name))
    #         params, n_params = get_function_paramaters(
    #             self.emission_comp[emission_name]['function'])
    #         self.emission_comp[emission_name]['param_names'] = params[2:]
    #         self.emission_comp[emission_name]['n_params'] = n_params - 2
    #
    #         self.emission_comp[emission_name]['params'] = {}
    #         self.emission_comp[emission_name]['component_bounds'] = {}
    #         for n in bounds_dict[emission_name]:
    #             self.emission_comp[emission_name]['params'][n] = bounds_dict[emission_name][n]
    #             self.emission_comp[emission_name]['params'][n]['arglist'] = [self.emission_comp[emission_name]['params'][n][i] for i in self.emission_comp[emission_name]['param_names']]
    #
    #     def emission_bounds(nu, beam, args=bounds_dict):
    #         Sv = 0
    #         for emission_name in self.emission_comp:
    #             for n in self.emission_comp[emission_name]['params']:
    #                 args = [self.emission_comp[emission_name]['params'][n][i]
    #                         for i in self.emission_comp[emission_name]['param_names']]
    #                 Sv += self.emission_comp[emission_name]['function'](
    #                     nu, beam, *args)
    #         return Sv
    #     self.emissionbounds = emission_bounds
