#!/usr/bin/env python3
# ===============================================================================
# SciptName     : srcClass.py
# ScriptDir     : /src/core/srcClass.py
# Author(s)     : T.J.Rennie
# Description   : astroSrc class for holding astronomical source data and
#                 permitting easy modeling and parameter retrieval
# ===============================================================================

# P-I: IMPORTS
from astropy.coordinates import SkyCoord
import h5py
import inspect
import numpy as np
from pydoc import locate

from mapext.core.usefulFunctions import h5PullDict

# I: PROCESS CLASS


class astroSrc():

    '''
    Class for holding astronomical objects within MAPEXT runs and functions
    '''

    def __init__(self,
                 coord=None, rad=None, name=None, referance=None,
                 fluxes=None, component_fluxes=None,
                 src_models=None,
                 notes=None,
                 test=False):
        self.COORD = coord
        self.RAD = rad
        if name != None:
            self.NAME = name
        else:
            if self.COORD.b.degree < 0:
                sign = '-'
            else:
                sign = '+'
            self.NAME = 'G{:06.2f}{}{:05.2f}'.format(
                self.COORD.l.degree, sign, abs(self.COORD.b.degree))
        self.REFS = referance

        if fluxes != None:
            self.flux = fluxes
        else:
            self.flux = np.zeros((0)).astype([('method', 'S50'), ('survey', 'S50'), (
                'name', 'S50'), ('nu', float), ('Sv', float), ('Sv_e', float)])

        if component_fluxes != None:
            self.comp_flux = component_fluxes
        else:
            self.comp_flux = np.zeros((0)).astype(
                [('component', 'S50'), ('name', 'S50'), ('nu', float), ('Sv', float), ('Sv_e', float)])

        if src_models != None:
            self.src_model = src_models
        else:
            self.src_model = np.zeros((0)).astype([('survey', 'S50'), ('name', 'S50'), (
                'nu', float), ('params', float, (6)), ('params_e', float, (6))])

        self.NOTE = notes

    def load_src(self, srcfile=None):
        if srcfile == None:
            srcfile = 'SRCHDF5/'+self.ID
        f = h5py.File(srcfile, 'r')
        params = h5PullDict(f)
        f.close()
        self.COORD = f['general/COORD']
        self.NAME = f['general/NAME']
        self.add_flux(f['fluxes/total_flux'])

    def update(self,
               coord=None, name=None, referance=None,
               fluxes=None, component_fluxes=None,
               src_models=None,
               notes=None,
               test=False):
        self.COORD = coord
        if name != None:
            self.NAME = name
        if referance != None:
            self.REFS = referance
        if fluxes != None:
            self.add_flux(fluxes)
        if component_fluxes != None:
            self.comp_flux = component_fluxes
        if src_models != None:
            self.src_model = src_models
        if notes != None:
            self.NOTE = notes

    def add_src_model(self, data):
        data = np.array(data, dtype=[('survey', 'S50'), ('name', 'S50'), (
            'nu', float), ('params', float, (6)), ('params_e', float, (6))])
        self.src_model = np.concatenate((self.src_model, data))

    def add_flux(self, data):
        data = np.array(data, dtype=[('method', 'S50'), ('survey', 'S50'), (
            'name', 'S50'), ('nu', float), ('Sv', float), ('Sv_e', float)])
        self.flux = np.concatenate((self.flux, data))

    def add_compflux(self, data):
        data = np.array(data, dtype=[
                        ('component', 'S50'), ('name', 'S50'), ('nu', float), ('Sv', float), ('Sv_e', float)])
        self.flux = np.concatenate((self.flux, data))

    def return_dictionary(self):
        out_dict = {'general':    {'NAME':         self.NAME,
                                   'COORD':        [self.COORD.l.degree,
                                                    self.COORD.b.degree],
                                   'REFS':         self.REFS,
                                   'RAD':         self.RAD},
                    'fluxes':     {'total_flux':   self.flux,
                                   'comp_flux':    self.comp_flux},
                    'models':     {'src_model':    self.src_model},
                    'note':        self.NOTE}
        return out_dict

    # def change_model(self,modelname):
    #     if modelname in list(self.model_alt.keys()):
    #         self.def_model(self.model_alt[modelname])
    #
    # def def_model(self, model_dict):
    #     '''
    #     Function to set the model used to define the emission of the source in
    #     frequency space. Set in two parts: emission model and emission parameter
    #
    #     INPUTS:
    #     model_dict  : Dictionary of the form
    #            {EMISSION_TYPE:{n:{PARAMTER_NAME : PARAMETER_VALUE,...},...},...}
    #     '''
    #     # reset model
    #     self.emissionModel = 0
    #     self.emission_comp = {}
    #
    #     # # define model
    #     # from src.core.emissionMech import models_available as allowed_models
    #
    #     def get_function_paramaters(func):
    #         from inspect import getargspec
    #         params = getargspec(func).args
    #         nparams = len(params)
    #         return params, nparams
    #
    #     for emission_name in model_dict:
    #         # if emission_name not in allowed_models:
    #         #     print('WARNING: emission model {} not recognised from emission file'.format(
    #         #         emission_name))
    #         #     return
    #         self.emission_comp[emission_name] = {}
    #         self.emission_comp[emission_name]['nmod'] = np.max(
    #             len(list(model_dict[emission_name])))
    #         self.emission_comp[emission_name]['function'] = locate(
    #             'src.core.emissionMech.{}'.format(emission_name))
    #         params, n_params = get_function_paramaters(
    #             self.emission_comp[emission_name]['function'])
    #         self.emission_comp[emission_name]['param_names'] = params[2:]
    #         self.emission_comp[emission_name]['n_params'] = n_params - 2
    #
    #         self.emission_comp[emission_name]['params'] = {}
    #         self.emission_comp[emission_name]['component_models'] = {}
    #         for n in model_dict[emission_name]:
    #             self.emission_comp[emission_name]['params'][n] = model_dict[emission_name][n]
    #             self.emission_comp[emission_name]['params'][n]['arglist'] = [self.emission_comp[emission_name]['params'][n][i] for i in self.emission_comp[emission_name]['param_names']]
    #
    #     def emission_model(nu, beam, args=model_dict):
    #         Sv = 0
    #         for emission_name in self.emission_comp:
    #             for n in self.emission_comp[emission_name]['params']:
    #                 args = [self.emission_comp[emission_name]['params'][n][i]
    #                         for i in self.emission_comp[emission_name]['param_names']]
    #                 Sv += self.emission_comp[emission_name]['function'](
    #                     nu, beam, *args)
    #         return Sv
    #     self.emissionModel = emission_model
