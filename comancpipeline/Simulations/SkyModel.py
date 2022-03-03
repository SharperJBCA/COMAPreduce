# Sky Model for COMAP pipeline simulations
#
import numpy as np
from comancpipeline.Simulations import Models

class SkyModel:

    def __init__(self, model_params):
        """

        All models in model info will be summed together when
        return the TOD

        model_params = {'model_name(sub_name)':
                      {'mapfile':'/path/to/map.fits',
                       'frequency_model':'func_in_frequency_models.py',
                       'fmodel_param1':...}
                       'model_name2':...
                      }
        """
        self.models = {}
        for model_name, model_info in model_params.items():
            class_name = model_name.split('(')[0]
            comp_name  = model_name.split('(')[-1][:-1]
            self.models[comp_name] = Models.__dict__[class_name](**model_info)


    def __call__(self, gl, gb, frequency):
        """
        Always work in galactic frame for now, allow for 
        rotations in future
        """

        tod = np.zeros(gl.size)
        for comp_name, model in self.models.items():
            tod += model(gl,gb,frequency)
        return tod
