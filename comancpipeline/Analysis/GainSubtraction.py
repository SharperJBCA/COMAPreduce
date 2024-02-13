#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  13 12:45:03 2024

Moved all of the gain subtraction code to its own module for easier access and modification. 

@author: sharper
"""

###
# Code for solving the gain solution
###
import numpy as np 
import logging 
from scipy.sparse.linalg import LinearOperator, bicg, cg, gmres
class AMatrix: 

    def __init__(self, templates, white_noise, fknee, alpha, tau, use_prior=False):
        self.templates = templates
        self.white_noise = white_noise
        self.fknee = fknee
        self.alpha = alpha
        self.tau = tau
        self.use_prior = use_prior

    def z_operation(self, d, template):
        """
        Difference between sum of each channel and the template weighted sum of each channel

        Arguments:
        ----------
        d - length frequency*time 
        template - length frequency 
        """
        n_tod = d.size//template.shape[0]
        data = d.reshape((n_tod, template.shape[0])).T
        C = template.T.dot(template)
        if not np.isfinite(np.sum(C)):
            logging.info('C matrix is not finite')
            logging.info('Nansum templates: {}'.format(np.nansum(template)))
            raise ValueError('C matrix is not finite in AMatrix.z_operation')
        TT = np.linalg.inv(C)

        d_sub = template.dot(TT.dot(template.T @ data)) 

        #top = np.sum(d.reshape((d.size//template.size, template.size)) * template, axis=1) 
        #bottom = np.sum(template**2)
        #average =  top / bottom 
        d_z = d - d_sub.T.flatten() 
        #np.repeat(average, template.size) * np.tile(template, d.size//template.size)  

        return d_z

    def p_operation(self, g, template):
        """
        Multiply the template by the template and stretch to frequency*time length 

        Arguments:
        ----------
        g - length time
        template - length frequency 
        """
        return np.repeat(g, template.size) * np.tile(template, g.size)  

    def p_transpose_operation(self, d, template):
        """
        Sum the data along frequency axis and multiply by template

        Arguments:
        ----------
        d - length frequency*time
        template - length frequency 
        """
        N = d.size//template.size
        Ptd = np.sum(d.reshape((N, template.size)) * template[np.newaxis,:], axis=1)

        return Ptd

    def create_power_spectrum(self, white_noise, fknee, alpha, sample_rate, n_samples):
        """
        Create a power spectrum for the 1/f noise

        Arguments:
        ----------
        white_noise - white noise level [K]
        fknee - knee frequency [Hz]
        alpha - spectral index
        sample_rate - sample rate [Hz]
        n_samples - number of samples
        """
        freqs = np.fft.fftfreq(n_samples, d=1./sample_rate) 
        power_spectrum = white_noise**2 * (np.abs(freqs / fknee)**alpha) 
        power_spectrum[0] = power_spectrum[-1]
        return freqs, power_spectrum

    def covariance_operation(self, g, white_noise, fknee, alpha):
        """
        Create the covariance matrix for the 1/f noise
        Assumes circulant structure and that 1/power_spectrum is the inverse of the covariance matrix

        Arguments:
        ----------
        g - length time
        white_noise - white noise level [K]
        fknee - knee frequency [Hz]
        alpha - spectral index
        """
        power_spectrum = self.create_power_spectrum(white_noise, fknee, alpha, sample_rate=1./self.tau, n_samples=g.size)[1]
        fft_g = np.fft.fft(g) 
        fft_Cg = fft_g/power_spectrum
        Cg = np.fft.ifft(fft_Cg).real 
        return Cg, power_spectrum

    def create_b(self, d):
        """
        Create the b vector for the gain solution

        Arguments:
        ----------
        d - length frequency*time
        """
        N_templates = self.templates.shape[1] 
        Zd = self.z_operation(d, self.templates[:,:N_templates-1]) 
        PtZd = self.p_transpose_operation(Zd, self.templates[:,N_templates-1]) 
    
        return PtZd

    def __call__(self,g):
        N_templates = self.templates.shape[1] 
        Pg = self.p_operation(g,self.templates[:,N_templates-1]) # stretch vector to be length of time*frequencies
        ZPg = self.z_operation(Pg, self.templates[:,:N_templates-1]) # subtract weighted mean off each time step 
        PtZPg = self.p_transpose_operation(ZPg, self.templates[:,N_templates-1]) # bin data into vector of length time, sum over frequency 

        if np.isnan(np.sum(PtZPg)):
            raise ValueError('PtZPg is not finite in AMatrix.__call__') 
        
        if self.use_prior:
            Cg = self.covariance_operation(g, self.white_noise, self.fknee, self.alpha)[0]
            return PtZPg + Cg
        else:
            return PtZPg 

def solve_gain_solution(d, templates, white_noise, fknee, alpha, tau=1./50., use_prior=False):
    """
    This is the latest version of the gain subtraction algorithm where the gain is solved for using a linear operator and 
    the conjugate gradient method. 

    Arguments
    ---------
    d : array of shape (frequency*time) - observed data as a single vector
    templates : array of shape (frequency*time, 2) - templates of 1/Tsys and 1 (plus additional templates if needed)
    white_noise : float - white noise level of the data [K] - for 1/f noise prior 
    fknee : float - knee frequency of the 1/f noise prior [Hz]
    alpha : float - spectral index of the 1/f noise prior
    tau : float - time constant of the 1/f noise prior [s] (default: 1/50. s)
    """
    n = d.size//templates.shape[0]
    matvec = AMatrix(templates, white_noise, fknee, alpha, tau, use_prior=use_prior)
    A = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    b = matvec.create_b(d)

    try:
        g, info = cg(A, b)
    except ValueError: 
        g = np.zeros(n)
        info = 1
    return g 

def gain_subtraction_fit(data_normed : np.ndarray, system_temperature : np.ndarray, ps_fits : np.ndarray):
    """ 
    THIS IS THE FUNCTION CALLED IN Level1AveragingGainCorrection

    data_normed (n_bands, n_channels, n_tod)
    system_tempareture (n_bands, n_channels)

    ps_fits - Not used 

    This function prepares the templates (A in Ax = b) and the b vector before calling solve_gain_solution
    """ 

    # Prepare the data and mask bad channels 
    n_bands, n_channels, n_tod = data_normed.shape

    templates = np.ones((n_bands, n_channels, 3))
    n_templates = templates.shape[-1]
    v = np.linspace(-1,1,1024*4).reshape((4,1024))
    templates[..., 0] = 1./system_temperature  
    templates[..., 1] = v/system_temperature     
    end_cut = 20

    # Remove edge frequencies and the bad middle frequency
    bad_values = np.isnan(system_temperature)
    templates[:, :end_cut ,:] = 0
    templates[:, -end_cut:,:] = 0
    templates[:, 512-5:512+5 ,:] = 0
    templates[bad_values,:] = 0
    data_normed[:, :end_cut ,:] = 0
    data_normed[:, -end_cut:,:] = 0
    data_normed[:, 512-5:512+5 ,:] = 0
    data_normed[bad_values,:] = 0 

    templates = templates.reshape((n_bands * n_channels, n_templates)) 
    data_reshape = data_normed.reshape((n_bands * n_channels, n_tod))
    if np.sum(bad_values) == bad_values.size:
        return np.zeros(data_reshape.T.flatten().shape)

    dG = solve_gain_solution(data_reshape.T.flatten(), templates, *ps_fits, tau=1./50., use_prior=False)
    return dG



def fit_gain_fluctuations(y_feed : np.ndarray ,
                            tsys : np.ndarray , 
                            sigma0_prior : float, 
                            fknee_prior : float, 
                            alpha_prior : float):
    """
    THIS IS THE FUNCTION CALLED IN Level1AveragingGainCorrection (note used in this version)

    This is the implementation of the gain subtraction algorithm. It is a bit too slow for the pipeline
    but it useful for testing and understanding. 
    This is the original implementation of the gain subtraction algorithm from Havard. 

    Model: y(t, nu) = dg(t) + dT(t) / Tsys(nu) + alpha(t) / Tsys(nu) (nu - nu_0) / nu_0, nu_0 = 30 GHz
    """
    
    def model_prior(P,x):
        return P[1] * np.abs(x/1)**P[2]

    def gain_temp_sep(y, P, F, sigma0_g, fknee_g, alpha_g, samprate=50):
        
        freqs = np.fft.rfftfreq(len(y[0]), d=1.0/samprate)
        freqs[0] = freqs[1]
        n_freqs, n_tod = y.shape
        Cf = model_prior([sigma0_g**2, fknee_g, alpha_g], freqs)
        Cf[0] = 1
        
        N = y.shape[-1]//2 * 2
        sigma0_est = np.std(y[:,1:N:2] - y[:,0:N:2], axis=1)/np.sqrt(2)
        sigma0_est = np.mean(sigma0_est)
        Z = np.eye(n_freqs, n_freqs) - P.dot(np.linalg.inv(P.T.dot(P))).dot(P.T)
        
        RHS = np.fft.rfft(F.T.dot(Z).dot(y))
    
        z = F.T.dot(Z).dot(F)
        a_bestfit_f = RHS/(z + 2*sigma0_est**2/Cf)
        a_bestfit = np.fft.irfft(a_bestfit_f, n=n_tod)
        yz = y*1 
        yz[yz ==0] = np.nan 
        yz = np.nanmedian(yz,axis=0) 
        
        m_bestfit = np.linalg.inv(P.T.dot(P)).dot(P.T).dot(y - F*a_bestfit)

        return a_bestfit, m_bestfit   
    
    Nbands = 4
    Nfreqs = Nbands * 1024 # number of frequency channels
    scaled_freqs = np.linspace(-4.0 / 30, 4.0 / 30, Nfreqs)  # (nu - nu_0) / nu_0
    scaled_freqs = scaled_freqs.reshape((Nbands, Nfreqs/Nbands))
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
    y_feed[np.isnan(y_feed)] =0     
    
    # Reshape to flattened grid
    P = P.reshape((4 * Nfreqs, 2))
    F = F.reshape((4 * Nfreqs, 1))
    y_feed = y_feed.reshape((4 * Nfreqs, Ntod))

    # Fit dg, dT and alpha
    a_feed, m_feed = gain_temp_sep(y_feed, P, F, sigma0_prior, fknee_prior, alpha_prior)
    dg = a_feed[0]
    dT = m_feed[0]
    alpha = m_feed[1]

    return np.reshape(F*dg[None,:],(4,Nfreqs,dg.size))
