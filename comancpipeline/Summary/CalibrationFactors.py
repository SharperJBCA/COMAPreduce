# CalibrationFactors.py
# 
# Read in calibration files and summarise them for Data module.
#
# Output format: 
#  {Source}/obsid
#  {Source}/gain
#
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from comancpipeline.Analysis import BaseClasses 
from comancpipeline.Tools import General, CaliModels

_kb = 1.38e-23
_c  = 3e8

class CalibrationGains(BaseClasses.DataStructure):

    def __init__(self,**kwargs):
        super().__init__()
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __call__(self):
        
        # Step 1: Get the files 
        filelist = glob(f'{self.calibration_directory}/{self.calibration_prefix}_*.hd5')

        # Convert to level 2 path
        filelist = [self.level2_directory+'/'+f.split(self.calibration_prefix+'_')[-1] for f in filelist]

        # Step 2: Create output data structure
        output = {}
        nobs   = {}
        for ifile,filename in enumerate(tqdm(filelist)):
            data = h5py.File(filename,'r')
            source  = self.getSource(data)
            if not source in nobs:
                nobs[source] = 0
            nobs[source] += 1
            data.close()

        # Step 3: Loop over files, get source name and obs id
        source_count = {k:0 for k in nobs.keys()}
        for ifile,filename in enumerate(tqdm(filelist)):
            data = h5py.File(filename,'r')

            fshape = data['level2']['frequency'].shape
            frequency = np.mean(np.reshape(data['level2']['frequency'],(fshape[0],2,fshape[1]//2)),axis=-1)
            source  = self.getSource(data)
            comment = self.getComment(data)
            feeds, feed_indices, feed_dict= self.getFeeds(data,'all')
            if not source in output:
                output[source] = {'Values':{},'Errors':{}}
            # Read calibration

            for mode in ['Values','Errors']:
                for k,v in data['level2'][self.calibration_prefix][f'Gauss_Narrow_{mode}'].items():
                    if not k in output[source][mode]:
                        output[source][mode][k] = np.zeros([nobs[source],20]+list(v.shape[1:]))
                    output[source][mode][k][source_count[source],feed_indices,...] = v[:,...]
                    
                if not 'flux' in output[source][mode]:
                    output[source][mode]['flux'] = np.zeros([nobs[source],20]+list(v.shape[1:]))

                conv = 2*_kb * (frequency*1e9/_c)**2 * 1e26 
                if not 'sigy' in output[source][mode]:
                    beam = 1.13*output[source][mode]['sigx'][source_count[source]]**2*\
                           output[source][mode]['sigy_scale'][source_count[source]]*\
                           (np.pi/180.)**2 * 2.355**2
                    output[source][mode]['flux'][source_count[source]] = 2*np.pi*output[source][mode]['A'][source_count[source]]*\
                                                                         output[source][mode]['sigx'][source_count[source]]**2*\
                                                                         output[source][mode]['sigy_scale'][source_count[source]]*\
                                                                         (np.pi/180.)**2 * conv[None,...] #* beam
                else:
                    beam = 1.13*output[source][mode]['sigx'][source_count[source]]*\
                           output[source][mode]['sigy'][source_count[source]]*\
                           (np.pi/180.)**2 * 2.355**2
                    output[source][mode]['flux'][source_count[source]] = 2*np.pi*output[source][mode]['A'][source_count[source]]*\
                                                                         output[source][mode]['sigx'][source_count[source]]*\
                                                                         output[source][mode]['sigy'][source_count[source]]*\
                                                                         (np.pi/180.)**2* conv[None,...] 
                #print(output[source][mode]['A'][source_count[source],0,0,0],output[source][mode]['sigy'][source_count[source],0,0,0]*60)
                #print(output[source][mode]['flux'][source_count[source],0,0,0])
                # Read pointing
                for k in ['x0','y0']:
                    v = data['level2'][self.calibration_prefix][f'Gauss_Average_{mode}'][k]
                    if not k in output[source][mode]:
                        output[source][mode][k] = np.zeros([nobs[source],20])
                    output[source][mode][k][source_count[source],feed_indices] = v[:]

            if not 'obsid' in output[source]:
                output[source]['obsid'] = np.zeros([nobs[source]])
                output[source]['MJD'] = np.zeros([nobs[source]])

            output[source]['obsid'][source_count[source]] = int(self.getObsID(data))
            output[source]['MJD'][source_count[source]]   = self.getMJD(data)
            output[source]['frequency'] = frequency

            data.close()
            source_count[source] += 1
        # Step 4: Save to file
        self.write(output)

    def write(self,output):
        General.save_dict_hdf5(output,
                               f'{self.calibration_directory}/{self.calibration_output_filename}')


        if self.save_to_repo:
            self.save_repo_data(f'{self.calibration_directory}/{self.calibration_output_filename}')

    def save_repo_data(self,calfile):
        """
        Save gains to the Data module in the repository
        """

        data = General.load_dict_hdf5(calfile)

        models = {'TauA':CaliModels.TauAFluxModel(),
                  'CasA':CaliModels.CasAFluxModel(),
                  'CygA':CaliModels.CygAFlux,
                  'jupiter':CaliModels.JupiterFlux}

        gains = {}
        for k,v in data.items():
            fshape = v['Values']['flux'].shape
            Nobs = v['Values']['flux'].shape[0]
            flux = np.reshape(v['Values']['flux'],(fshape[0],fshape[1],8))
            freq = v['frequency'].flatten()
            eflux = np.reshape(v['Errors']['flux'],(fshape[0],fshape[1],8))

            if not k.lower() in gains:
                gains[k.lower()] = {'gains':np.zeros((Nobs,fshape[1],8)),
                                    'errors':np.zeros((Nobs,fshape[1],8)),
                                    'obsids':v['obsid']}
            for ifreq in range(freq.size):
                model_flux = models[k](freq[ifreq],v['MJD'],return_jansky=True,allpos=True)
            
                gains[k.lower()]['gains'][:,:,ifreq]  = flux[...,ifreq]/model_flux[:,None]
                gains[k.lower()]['errors'][:,:,ifreq] = eflux[...,ifreq]/model_flux[:,None]


        General.save_dict_hdf5(gains,'/scratch/nas_comap1/sharper/COMAP/COMAPreduce/comancpipeline/gains.hd5')
