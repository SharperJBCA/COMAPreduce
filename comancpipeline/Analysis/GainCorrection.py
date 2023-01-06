import numpy as np
from matplotlib import pyplot
import h5py
import comancpipeline
from comancpipeline.Analysis import BaseClasses
from comancpipeline.Analysis.FocalPlane import FocalPlane
from comancpipeline.Analysis import SourceFitting
from comancpipeline.Tools import Coordinates, Types, stats, FileTools
from os import listdir, getcwd
from os.path import isfile, join
import grp
import shutil
import os

from tqdm import tqdm
from scipy.interpolate import interp1d
import datetime
from astropy.time import Time
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

from scipy.signal import find_peaks
from scipy.optimize import minimize
__level2_gain_version__ = 'v1'

def model_prior(P,x):

    return P[1] * np.abs(x/1)**P[2]


def gain_temp_sep(y, P, F, sigma0_g, fknee_g, alpha_g, samprate=50):
    freqs = np.fft.rfftfreq(len(y[0]), d=1.0/samprate)
    n_freqs, n_tod = y.shape
    Cf = model_prior([sigma0_g**2, fknee_g, alpha_g], freqs)
    Cf[0] = 1
    
    N = y.shape[-1]//2 * 2
    sigma0_est = np.std(y[:,1:N:2] - y[:,0:N:2], axis=1)/np.sqrt(2)
    sigma0_est = np.mean(sigma0_est)
    Z = np.eye(n_freqs, n_freqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
    
    RHS = np.fft.rfft(F.T.dot(Z).dot(y))

    z = F.T.dot(Z).dot(F)
    a_bestfit_f = RHS/(z + sigma0_est**2/Cf)
    a_bestfit = np.fft.irfft(a_bestfit_f, n=n_tod)

    m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)
    
    return a_bestfit, m_bestfit

def fit_gain_fluctuations(y_feed, tsys, sigma0_prior, fknee_prior, alpha_prior):
    """
    Model: y(t, nu) = dg(t) + dT(t) / Tsys(nu) + alpha(t) / Tsys(nu) (nu - nu_0) / nu_0, nu_0 = 30 GHz
    """

    nsb, Nfreqs, Ntod = y_feed.shape

    scaled_freqs = np.linspace(-4.0 / 30, 4.0 / 30, 4 * 1024)  # (nu - nu_0) / nu_0
    scaled_freqs = scaled_freqs.reshape((4, 1024))
    scaled_freqs[(0, 2), :] = scaled_freqs[(0, 2), ::-1]  # take into account flipped sidebands

    P = np.zeros((4, Nfreqs, 2))
    F = np.zeros((4, Nfreqs, 1))
    P[:, :,0] = 1 / tsys
    P[:, :,1] = scaled_freqs/tsys
    F[:, :,0] = 1

    end_cut = 100
    # Remove edge frequencies and the bad middle frequency
    y_feed[:, :4] = 0
    y_feed[:, -end_cut:] = 0
    P[:, :4] = 0
    P[:, -end_cut:] = 0
    F[:, :4] = 0
    F[:, -end_cut:] = 0
    F[:, 512] = 0
    P[:, 512] = 0
    y_feed[:, 512] = 0

    calibrated = y_feed * tsys[:, :, None]  # only used for plotting
    calibrated[(0, 2), :] = calibrated[(0, 2), ::-1]

    # Reshape to flattened grid
    P = P.reshape((4 * Nfreqs, 2))
    F = F.reshape((4 * Nfreqs, 1))
    y_feed = y_feed.reshape((4 * Nfreqs, Ntod))

    # Fit dg, dT and alpha
    a_feed, m_feed = gain_temp_sep(y_feed, P, F, sigma0_prior, fknee_prior, alpha_prior)
    dg = a_feed[0]
    dT = m_feed[0]
    alpha = m_feed[1]

    spec = P.dot(m_feed)
    return np.reshape(F*dg[None,:],(4,Nfreqs,dg.size)), dT, alpha, spec

def model(P,x):
    """
    Assuming model \sigma_w^2 + \sigma_r^2 (frequency/frequency_r)^\alpha

    Parameters
    ----------

    Returns
    -------
    
    """
    return P[0] + P[1]*np.abs(x/1.)**P[2]

def error(P,x,y,sig2):

    chi2 = np.sum((np.log(y)-np.log(model([sig2,P[0],P[1]],x)))**2)
    
    return chi2

def fit_power(nu,P):

    nbins = 15
    nu_edges = np.logspace(np.log10(np.min(nu)),np.log10(np.max(nu)),nbins+1)
    top = np.histogram(nu,nu_edges,weights=P)[0]
    bot = np.histogram(nu,nu_edges)[0]
    nu_bin = np.histogram(nu,nu_edges,weights=nu)[0]/bot
    P_bin = top/bot

    gd = (bot != 0) & np.isfinite(P_bin)
    nu_bin = nu_bin[gd]
    P_bin = P_bin[gd]
    if len(nu_bin) == 0:
        raise IndexError

    P0 = [P_bin[np.argmin((nu_bin-1)**2)], -1]
    gd = (nu_bin > 0.1) # just the high frequencies
    result = minimize(error,P0,args=(nu_bin[gd],P_bin[gd],P_bin[-1]),bounds=([0,None],[None,0]))

    results = [P_bin[-1], result.x[0],result.x[1]]
    return results, nu_bin, P_bin

class RepointEdges(BaseClasses.DataStructure):
    """                                                                                                                                                         
    Scan Edge Split - Each time the telescope stops to repoint this is defined as the edge of a scan                                                            
    """

    def __init__(self, **kwargs):

        self.scan_status_code = 1
        for item, value in kwargs.items():
            self.__setattr__(item,value)

    def __call__(self, data, source=''):
        """                                                                                                                                                    
        Expects a level 1 data structure                                                                                                                       
        """
        
        return self.getScanPositions(data)

    def getScanPositions(self, d):
        """                                                                                                                                                     
        Finds beginning and ending of scans, creates mask that removes data when the telescope is not moving,                                                   
        provides indices for the positions of scans in masked array                                                                                             
                                                                                                                                                                
        Notes:                                                                                                                                                  
        - We may need to check for vane position too                                                                                                            
        - Iteratively finding the best current fraction may also be needed                                                                                      
        """
        features = self.getFeatures(d)
        scan_status = d['hk/antenna0/deTracker/lissajous_status'][...]
        scan_utc    = d['hk/antenna0/deTracker/utc'][...]
        scan_status_interp = interp1d(scan_utc,scan_status,kind='previous',bounds_error=False,
                                      fill_value='extrapolate')(d['spectrometer/MJD'][...])

        scans = np.where((scan_status_interp == self.scan_status_code))[0]
        diff_scans = np.diff(scans)
        edges = scans[np.concatenate(([0],np.where((diff_scans > 1))[0], [scans.size-1]))]
        scan_edges = np.array([edges[:-1],edges[1:]]).T

        return scan_edges

class CreateLevel2GainCorr(BaseClasses.DataStructure):
    """
    Takes level 1 files, bins and calibrates them for continuum analysis.
    """

    def __init__(self, feeds='all',
                 output_obsid_starts = [0],
                 output_obsid_ends   = [None],
                 output_dirs = ['.'],
                 nworkers= 1,
                 database=None,
                 average_width=512,
                 calvanedir='CalVanes',
                 cal_mode = 'Vane',
                 cal_prefix='',level2='level2',
                 data_dirs=None,
                 set_permissions=True,
                 permissions_group='comap',
                 calvane_prefix='CalVane',**kwargs):
        """
        nworkers - how many threads to use to parallise the fitting loop
        average_width - how many channels to average over
        """
        super().__init__(**kwargs)


        self.name = 'CreateLevel2Cont'
        self.feeds_select = feeds

# We will be writing level 2 data to multiple drives,
        #  drive to write to will be set by the obsid of the data
        self.output_dirs = output_dirs
        self.output_obsid_starts = output_obsid_starts
        self.output_obsid_ends   = output_obsid_ends

        if isinstance(data_dirs,list):
            self.data_dirs = data_dirs
        else:
            self.data_dirs = [data_dirs]

        self.nworkers = int(nworkers)
        self.average_width = int(average_width)

        self.calvanedir = calvanedir
        self.calvane_prefix = calvane_prefix

        self.cal_mode = cal_mode
        self.cal_prefix=cal_prefix

        self.level2=level2
        self.set_permissions = set_permissions
        self.permissions_group = permissions_group

        self.database   = database + '_{}'.format(os.getpid())

    def __str__(self):
        return ""

    def __call__(self,data):
        """
        """
        #print(type(data))
        assert isinstance(data, h5py._hl.files.File), 'Data is not a h5py file structure'
        fname = data.filename.split('/')[-1]
        self.logger(f' ')
        self.logger(f'{fname}:{self.name}: Starting. (overwrite = {self.overwrite})')


        self.obsid = self.getObsID(data)
        self.comment = self.getComment(data)
        prefix = data.filename.split('/')[-1].split('.hd5')[0]

        self.output_dir = self.getOutputDir(self.obsid,
                                            self.output_dirs,
                                            self.output_obsid_starts,
                                            self.output_obsid_ends)
        self.calvanepath= '{}/{}'.format(self.output_dir,self.calvanedir)

        data_dir_search = np.where([os.path.exists('{}/{}_Level2Cont.hd5'.format(data_dir,prefix)) for data_dir in self.data_dirs])[0]
        if len(data_dir_search) > 0:
            self.output_dir = self.data_dirs[data_dir_search[0]]
        self.outfilename = '{}/level3_{}.hd5'.format(self.output_dir,prefix)

        # Skip files that are already calibrated:
        #print(self.overwrite, os.path.exists(self.outfilename),self.outfilename)
        if os.path.exists(self.outfilename) & (not self.overwrite):
            self.outfile = h5py.File(self.outfilename,'r')
            data.close()
            return self.outfile

        self.logger(f'{fname}:{self.name}: Applying vane calibration. Bin width {self.average_width}.')
        self.run(data)
        self.logger(f'{fname}:{self.name}: Writing level 2 file: {self.outfilename}')
        self.write(data)

        if data:
            data.close()
        return self.output
        #return self.outfile


    def run(self,data):
        """
        Read in the level 2 data, apply the Oslo gain correction
        """

        script_dir = os.path.dirname(__file__)
        prior = h5py.File(f'{script_dir}/Cf_prior_data.hdf5','r')
        alpha_prior = prior['alpha_prior'][...]
        sigma0_prior= prior['sigma0_prior'][...]
        fknee_prior = prior['fknee_prior'][...]
        prior.close()

        # get the start/end points of the vane
        vane_filename = '{}/{}_{}'.format(self.calvanepath,self.calvane_prefix,os.path.basename(data.file.filename))
        vane = h5py.File(vane_filename,'r')
        vane_null,vane_first = int(vane['VaneEdges'][0,0]), int(vane['VaneEdges'][0,1])
        if vane['VaneEdges'].shape[0] == 2:
            vane_second = int(vane['VaneEdges'][1,0])
        else:
            vane_second = None
        tsys = vane['Tsys'][0,...]
        gain = vane['Gain'][0,...]
        scan_edge_obj = RepointEdges()
        #if 'TauA' in data['comap'].attrs['comment']:
        #print(data['comap'].attrs['comment'])
        #scan_edges = np.array([[vane_first, vane_second]])
        #else:
        try:
            scan_edges = scan_edge_obj(data)
        except IndexError:
            scan_edges = np.array([[vane_first, vane_second]])

        # loop over feeds
        tod_grp = data['spectrometer/tod']
        el_grp  = data['spectrometer/pixel_pointing/pixel_el']
        ra_grp  = data['spectrometer/pixel_pointing/pixel_ra']
        dec_grp  = data['spectrometer/pixel_pointing/pixel_dec']
        feature_grp=data['spectrometer/features']
        n_feeds,n_bands,n_channels,n_tod = data['spectrometer/tod'].shape

        feeds = data['spectrometer/feeds'][:]
        freqs = data['spectrometer/frequency'][:]
        lowf = 0
        highf= freqs.shape[1]
        from scipy.sparse import block_diag, linalg
        #fig1 = pyplot.figure(1,figsize=(12,8))
        #fig2 = pyplot.figure(2,figsize=(12,8))

        tod_ps_all = np.zeros((20,tod_grp.shape[1],tod_grp.shape[3])) # Contains the averaged data
        tod_ps_orig= np.zeros((20,tod_grp.shape[1],tod_grp.shape[3])) # Contains the unfiltered data
        wei_ps_all = np.zeros((20,tod_grp.shape[1],tod_grp.shape[3]))
        fnoise_stats = np.zeros((n_feeds,4,len(scan_edges),3))
        dT_all = np.zeros((20,tod_grp.shape[3]))
        alpha_all = np.zeros((20,tod_grp.shape[3]))

        
        # Bright sources list:
        bright_sources = [[30.75,-0.04,7./60.]]

        for ifeed in range(n_feeds):
            obsid = os.path.basename(data.file.filename).split('-')[1]
            fig_dir = f'test_processing_scripts/figures/{obsid}'
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for iscan,(start, end) in enumerate(tqdm(scan_edges[:,:])):
                tod = tod_grp[ifeed,...,start:end]
                
                ra  = ra_grp[ifeed,start:end]
                dec =dec_grp[ifeed,start:end]
                gl,gb = Coordinates.e2g(ra,dec)
                for (xsrc,ysrc,rsrc) in bright_sources:
                    src_sep = Coordinates.AngularSeperation(xsrc,ysrc,gl,gb)
                    src_mask= (src_sep < rsrc)

                el  = el_grp[ifeed,start:end]
                features = np.unique(np.log(feature_grp[start:end])/np.log(2))
                tod_mean = np.mean(tod,axis=-1)
            
                A = 1./np.sin(el*np.pi/180.)
                Amasked = A[~src_mask]
                Nfreq = tod[-1,lowf:highf,0].size
                tod = tod/tod_mean[...,None] - 1 #/tod_mean[...,None] - 1 #(gain[ifeed]*tsys[ifeed])[...,None]
                #tod_mean[...,None] 
                Z = np.ones((A.size, 2))
                Z[:,1] = A
                Z = block_diag([Z]*Nfreq)
                Zmasked = np.ones((Amasked.size, 2))
                Zmasked[:,1] = Amasked
                Zmasked = block_diag([Zmasked]*Nfreq)
                
                
                # Remove the atmosphere fluctuations
                if not any([feature==5.0 for feature in features]):
                    tod_ps = np.zeros(tod.shape)
                    for i in range(tod.shape[0]):
                        d = tod[i,lowf:highf,~src_mask].flatten()[:,None]
                        b = Zmasked.T.dot(d)
                        M = (Zmasked.T.dot(Zmasked))
                        a = linalg.spsolve(M,b)
                        tod_ps[i] = tod[i,lowf:highf,:] - np.reshape(Z.dot(a[:,None]),(Nfreq,tod.shape[2]))
                else:
                    tod_ps = tod[:,lowf:highf,:]
                stepsize = 50
                N = tod_ps.shape[-1]//stepsize * stepsize
                tod_ps_std = np.ones(tod_ps.shape)
                atemp = np.repeat(np.nanstd(np.reshape(tod_ps[...,:N],(tod_ps.shape[0], 
                                                                       tod_ps.shape[1], 
                                                                       N//stepsize, stepsize)),axis=-1),stepsize,axis=-1)
                print(atemp.shape,tod_ps_std.shape)
                tod_ps_std[...,:N] = atemp
                tod_ps_std[...,N:] = tod_ps_std[...,N-1:N]
                
                fig = pyplot.figure()
                #pyplot.imshow(tod_ps_std[0,...],aspect='auto',vmax=0.0075)
                #pyplot.colorbar()
                pyplot.plot(np.nanmean(tod_ps_std,axis=-1).flatten())
                pyplot.yscale('log')
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_Scan{iscan:02d}_std.png')
                pyplot.close(fig)
                
                # Weights
                wei = (1e9/1024./50.)/tsys[ifeed]**2
                wei[tsys[ifeed] == 0] = 0
                wei[...,:100] = 0
                wei[...,-100:]= 0
                wei[...,510:514]  = 0
                # Get the averaged original data
                resid_orig = tod_ps*tsys[ifeed,:,:,None]
                resid_avg_orig = np.sum(resid_orig*wei[:,:,None],axis=1)/np.sum(wei,axis=1)[:,None]
                tod_ps_orig[feeds[ifeed]-1,...,start:end] = resid_avg_orig

                fig = pyplot.figure()
                pyplot.plot(resid_avg_orig[0],'k')
                pyplot.xlabel('Sample')
                pyplot.ylabel(r'$T_a$ (K)')
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_Scan{iscan:02d}_cal_avg_tod.png')
                pyplot.close(fig)
                
                # Calculate the power spectrum
                resid_orig = tod_ps
                resid_avg_orig = np.sum(resid_orig*wei[:,:,None],axis=1)/np.sum(wei,axis=1)[:,None]
                
                fig = pyplot.figure()
                pyplot.plot(resid_avg_orig[0],'k')
                pyplot.xlabel('Sample')
                pyplot.ylabel('DU')
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_Scan{iscan:02d}_uncal_avg_tod.png')
                pyplot.close(fig)
                
                ps = np.abs(np.fft.fft(resid_avg_orig[0])**2)
                ps_nu = np.fft.fftfreq(ps.size,d=1./50.)
                ps_N = ps.size//2
                try:
                    ps_fits, ps_nu_bin, ps_bin = fit_power(ps_nu[1:ps_N], ps[1:ps_N])
                except IndexError:
                    continue
                fig = pyplot.figure()
                pyplot.plot(ps_nu[1:ps_N],ps[1:ps_N])
                pyplot.plot(ps_nu_bin,ps_bin,'o')
                pyplot.plot(ps_nu_bin,model(ps_fits,ps_nu_bin))
                pyplot.xlabel('Frequency (Hz)')
                pyplot.ylabel(r'Power (K$^2$)')
                pyplot.grid()
                pyplot.xscale('log')
                pyplot.yscale('log')
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_fitted_spectrum_Scan{iscan:02d}.png')
                pyplot.close(fig)

                fnoise_stats[ifeed,:,iscan,:] = np.array([ps_fits[0],
                                                          ps_fits[1], 
                                                          ps_fits[2]])[None,:]
                # fit gain fluctuations
                dG, dT, alpha,spectrum_model = fit_gain_fluctuations(tod_ps, tsys[ifeed], 
                                                                     np.sqrt(ps_fits[0]), ps_fits[1], ps_fits[2])
                tod_cal = tod_ps*tsys[ifeed,:,:,None]

                # Get the averaged clean data
                resid = (tod_ps - dG)*tsys[ifeed,:,:,None]
                resid_avg = np.sum(resid*wei[:,:,None],axis=1)/np.sum(wei,axis=1)[:,None]
                tod_ps_all[feeds[ifeed]-1,...,start:end] = resid_avg
                wei_ps_all[feeds[ifeed]-1,...,start:end] = np.sum(wei,axis=1)[:,None]
                
                dT_all[feeds[ifeed]-1,start:end] = dT
                alpha_all[feeds[ifeed]-1,start:end]=alpha


                # Plot the spectrum for subtracted emission
                print(start,end)
                #if (start < 162985) & (end > 162985):
                print('HELLO')
                idx = np.argmax(tod_ps_orig[ifeed,0,start:end])
                print(idx)
                fig = pyplot.figure(figsize=(12,8))
                pyplot.subplot(211)
                pyplot.plot(tod_ps_orig[ifeed,0,start+idx-100:start+idx+100])
                pyplot.plot(tod_ps_all[ifeed,0,start+idx-100:start+idx+100])
                pyplot.subplot(212)
                pyplot.plot(tod_ps[:,:,idx].flatten())
                pyplot.plot(spectrum_model[:,idx].flatten())
                pyplot.ylabel('Normalised Brightness')
                pyplot.xlabel('Sample')
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_Scan{iscan:02d}_W43.png')
                pyplot.close(fig)
                
                # Make some plots of the power spectra
                cal_ps = np.abs(np.fft.fft(tod_cal[0,550,:]))**2
                dG_ps = np.abs(np.fft.fft(dG[0,550,:]*tsys[ifeed,0,550]))**2
                rd_ps = np.abs(np.fft.fft(resid[0,550,:]))**2
                rd_full_ps = np.abs(np.fft.fft(resid_avg[0,:]))**2
                dt_ps = np.abs(np.fft.fft(dT/tsys[ifeed,0,550]))**2
                nu_ps = np.fft.fftfreq(dG.shape[-1],d=1./50.)
                N = nu_ps.size//2
                
                # Plot power spectra comparisons
                fig = pyplot.figure(figsize=(12,8))
                pyplot.plot(nu_ps[1:N], cal_ps[1:N],label='Original')
                pyplot.plot(nu_ps[1:N], dG_ps[1:N],label='Gain Model')
                pyplot.plot(nu_ps[1:N], rd_ps[1:N],label='Residual',alpha=0.8)
                pyplot.plot(nu_ps[1:N], rd_full_ps[1:N],label='Residual Avg', alpha=0.8)
                pyplot.plot(nu_ps[1:N], model_prior(ps_fits,nu_ps[1:N])*tsys[ifeed,0,550]**2,
                            label='Model',color='k',lw=3)
                pyplot.yscale('log')
                pyplot.xscale('log')
                pyplot.legend()
                pyplot.savefig(f'{fig_dir}/Feed{feeds[ifeed]:02d}_Scan{iscan:02d}_gainfunction.png')
                pyplot.close(fig)
                print('Figure closed')
                        
        cov5s = np.zeros((20,20,len(scan_edges)))
        for iscan, (start,end) in enumerate(tqdm(scan_edges)):
            Nstep = int(250)
            tod_z = tod_ps_orig[:,0,start:end]
            N = tod_z.shape[-1]//Nstep * Nstep
            tod_5s = np.nanmean(np.reshape(tod_z[:,:N], (20, N//Nstep, Nstep)),axis=-1)
            tod_5s = (tod_5s - np.nanmean(tod_5s,axis=-1)[:,None])/np.nanstd(tod_5s,axis=-1)[:,None]
            cov5s[...,iscan] = tod_5s.dot(tod_5s.T)/tod_5s.shape[-1]

        cov5s_filt = np.zeros((20,20,len(scan_edges)))                              
        for iscan, (start,end) in enumerate(tqdm(scan_edges)):
            Nstep = int(250)
            tod_z = tod_ps_all[:,0,start:end]
            N = tod_z.shape[-1]//Nstep * Nstep
            tod_5s = np.nanmean(np.reshape(tod_z[:,:N], (20, N//Nstep, Nstep)),axis=-1)
            tod_5s = (tod_5s - np.nanmean(tod_5s,axis=-1)[:,None])/np.nanstd(tod_5s,axis=-1)[:,None]
            cov5s_filt[...,iscan] = tod_5s.dot(tod_5s.T)/tod_5s.shape[-1]


        pointing = [np.zeros((20,tod_ps_all.shape[-1])) for i in range(4)]
        self.data_out = {'feeds':data['spectrometer/feeds'][:],
                         'frequency':data['spectrometer/frequency'][...],
                         'scan_edges':scan_edges,
                         'weights':wei_ps_all,
                         'cov5s':cov5s,
                         'dT':dT_all,
                         'alpha':alpha_all,
                         'filtered_cov5s':cov5s_filt,
                         'fnoise_fits':fnoise_stats,
                         'filtered_tod':tod_ps_all,
                         'tod':tod_ps_orig}

        for k,p in zip(['az','el','ra','dec'],pointing):
            p[self.data_out['feeds'].astype(int)-1] = data[f'spectrometer/pixel_pointing/pixel_{k}'][...] 
            self.data_out[f'pixel_pointing/pixel_{k}'] = p
        self.data_out['MJD'] = data['spectrometer/MJD'][...]

    def write(self,data):
        """
        Write out as a mock level3 data.
        """
        data_dir = self.output_dir 

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        input_filename  = os.path.basename(data.file.filename)
        level3_filename = data_dir + '/level3_' + input_filename

        self.output = h5py.File(level3_filename,'w')

        data.copy('comap',self.output) 
        for k,v in self.data_out.items():
            
            if '/' in k:
                groupname, dsetname = k.split('/')
                if groupname in self.output:
                    grp = self.output[groupname]
                else:
                    grp = self.output.create_group(groupname)
                grp.create_dataset(dsetname, data=v)
            else:
                if k in self.output:
                    del self.output[k]
                self.output.create_dataset(k,data=v)


            
