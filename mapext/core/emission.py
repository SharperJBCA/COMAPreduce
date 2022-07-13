import numpy as np

# Synchrotron Emission
def synchrotron(nu, beam, A_sync, alpha):

    # TODO: add beam here so that amplitude is in Janksys
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34

    S = A_sync * nu**alpha
    return S


def synchrotron_2comp(nu, beam, A0, alp0, A1, alp1, d):
    # TODO: add beam here so that amplitude is in Janksys
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34

    x_b = (A0/A1)**(1/(alp1-alp0))
    A = A0 * x_b**alp0 * 2**((alp1-alp0)*d)
    S = A * (nu/x_b)**alp0 * (0.5*(1+(nu/x_b)**(1/d)))**((alp1-alp0)*d)
    return S

# Free-Free Emission
def freefree(nu, beam, EM):
    ''' Full Draine 2011 free-free model put into numpy '''

    Te = 7500 # K, fixed temperature

    c = 299792458.
    g_ff = np.log(np.exp(5.960 - (np.sqrt(3.0)/np.pi)*np.log(np.multiply(nu, (Te/10000.0)**(-3.0/2.0) ) )) + 2.71828)
    tau_ff = 5.468e-2 * np.power(Te,-1.5) * np.power(nu,-2.0) * EM * g_ff
    T_ff = Te * (1.0 - np.exp(-tau_ff))
    S = 2.0 * 1380.6488 * T_ff * np.power(np.multiply(nu,1e9),2) / c**2  * beam
    return S


# Anomalous Microwave Emission (Lognormal Approximation)
def ame(nu, beam, A_AME, nu_AME, W_AME):
    ''' Here the beam doesn't do anything, but you still need it '''

    # TODO: scale by the beam so that amplitude of AME is in janskys!!!
    nlog = np.log(nu)
    nmaxlog = np.log(nu_AME)
    S = A_AME*np.exp(-0.5 * ((nlog-nmaxlog)/W_AME)**2)
    return S


# Thermal Dust Emission
def thermaldust(nu, beam, T_d, tau, beta):
    c = 299792458.
    k = 1.3806488e-23
    h = 6.62606957e-34
    nu = np.multiply(nu,1e9)

    planck = np.exp(h*nu/k/T_d) - 1.
    modify = 10**tau * (nu/353e9)**beta # set to tau_353
    S = 2 * h * nu**3/c**2 /planck * modify * beam * 1e26
    return S
