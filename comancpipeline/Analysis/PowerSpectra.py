import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class FitPowerSpectrum:

    def __init__(self, nbins=15):
        self.nbins = nbins
        self.__result = None 

    @property 
    def result(self):
        return self.__result
    
    @result.setter
    def result(self, value):
        self.__result = value

    def bin_power_spectrum(self, freqs, power_spectrum, errors=1, min_freq=None, max_freq=None):
        """
        Bin the power spectrum into nbins

        Arguments:
        ----------
        freqs - frequency vector
        power_spectrum - power spectrum vector
        """
        if isinstance(min_freq, type(None)):
            min_freq = np.min(freqs)
        if isinstance(max_freq, type(None)):
            max_freq = np.max(freqs)
        # Bin the power spectrum
        nu_edges = np.logspace(np.log10(min_freq),np.log10(max_freq),self.nbins+1)
        top = np.histogram(freqs,nu_edges,weights=power_spectrum)[0]
        bot = np.histogram(freqs,nu_edges)[0]
        gd = (bot != 0)
        P_bin = np.zeros(bot.size) + np.nan
        nu_bin = np.zeros(bot.size) + np.nan
        nu_bin[gd] = np.histogram(freqs,nu_edges,weights=freqs)[0][gd]/bot[gd]
        P_bin[gd] = top[gd]/bot[gd]
        
        # Compute the power law fit
        gd = (bot != 0) & np.isfinite(P_bin) & (nu_bin != 0)
        nu_bin = nu_bin[gd]
        P_bin = P_bin[gd]
        return nu_bin, P_bin

    def knee_frequency_model(self, P, freqs):
        """
        Model for the knee frequency

        Arguments:
        ----------
        P - power spectrum
        freqs - frequency vector
        """
        sigma_white, knee, alpha = P 
        return sigma_white**2 * (1+np.abs(freqs / knee)**alpha)

    def red_noise_model(self, P, freqs):
        """
        Model for the knee frequency

        Arguments:
        ----------
        P - power spectrum
        freqs - frequency vector
        """
        sigma_white, red_noise, alpha = P 
        return sigma_white**2 + red_noise**2 * np.abs(freqs / 1)**alpha

    def knee_frequency_model_rolloff(self, P, freqs):
        sigma_white, knee, alpha = P 
        x = freqs - 0.5
        S = 10**(1- 1/(1 + np.exp(-x*10)))/100. + 0.9 
        return  sigma_white**2 * (1+np.abs(freqs / knee)**alpha)  * S 


    def sigma_red_model(self, P, freqs):
        """
        Model for the knee frequency

        Arguments:
        ----------
        P - power spectrum
        freqs - frequency vector
        """
        sigma_white, sigma_red, alpha = P 
        return sigma_white**2 + sigma_red**2 * np.abs(freqs / knee)**alpha

    def error(self, P, freqs, data, err, model):
        """
        Error function for the knee frequency

        Arguments:
        ----------
        P - power spectrum
        freqs - frequency vector
        data - data vector
        err - error vector
        """
        chi2 = np.sum((data - model(P, freqs))**2 / err**2) 

    def log_error(self, P, freqs, data, err, model):
        """
        Error function for the knee frequency

        Arguments:
        ----------
        P - power spectrum
        freqs - frequency vector
        data - data vector
        err - error vector
        """
        log_err = err/data 
        chi2 = np.sum((np.log(data) - np.log(model(P, freqs)))**2)# / log_err**2) 
        return chi2 

    def plot_fit(self,fig_dir='figures/',prefix=''):
        """
        Plot the fit
        """
        fig = plt.figure()
        plt.plot(self.nu_bin, self.P_bin, 'o')
        plt.plot(self.nu_bin, self.red_noise_model(self.result.x, self.nu_bin))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [K$^2$]')
        plt.text(0.95, 0.95, f'$\sigma_{{white}}$ = {self.result.x[0]:.2f} K)\n $\sigma_{{red}}$ = {self.result.x[1]:.2f} K)\n $\\alpha$ = {self.result.x[2]:.2f}', 
                        horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        plt.savefig(f'{fig_dir}/{prefix}power_spectrum_fit.png')
        plt.close(fig)

    def __call__(self, freqs, data, errors=None, model=None, error_func=None, P0=None, min_freq=None,max_freq=None): 
        if isinstance(errors, type(None)):
            errors = np.ones_like(data) 
        if isinstance(model, type(None)):
            model = self.knee_frequency_model
        if isinstance(error_func, type(None)):
            error_func = self.error

        self.nu_bin, self.P_bin = self.bin_power_spectrum(freqs, data, errors, min_freq=min_freq, max_freq=max_freq) 

        if len(self.nu_bin) < 3:
            self.result = None

        if isinstance(P0, type(None)):
            if model == self.knee_frequency_model:
                P0 = [self.P_bin[-1]**0.5, np.mean(self.nu_bin), np.log(self.P_bin[0]/self.P_bin[-1])/np.log(self.nu_bin[0]/self.nu_bin[-1])] 
            elif model == self.red_noise_model:
                idx = np.argmin((self.nu_bin - 1)**2)
                P0 = [self.P_bin[-1]**0.5, self.P_bin[idx]**0.5, np.log(self.P_bin[0]/self.P_bin[-1])/np.log(self.nu_bin[0]/self.nu_bin[-1])] 

        self.result = minimize(error_func, P0, method='L-BFGS-B',
                            args=(self.nu_bin, self.P_bin, 1, model), 
                            bounds=[(P0[0]*0.95,P0[0]*1.05), (0, None), (-10, 0)])

