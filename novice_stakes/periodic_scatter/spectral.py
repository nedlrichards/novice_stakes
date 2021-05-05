import numpy as np
from math import pi
from mpmath import mp
from scipy.special import factorial, erfc, kn

def define_ms(kcL, alpha_0L, dx_L, num_eva):
    """calculate a spectral sum vector based on number of evanescent terms"""
    kcL = np.real(kcL)
    num_p = np.fix((kcL - alpha_0L) / (2 * pi)) + num_eva
    num_n = np.fix(-(kcL + alpha_0L) / (2 * pi)) - num_eva
    ms = np.arange(num_n, num_p + 1)

    # compute wavenumbers
    alpha_mL = alpha_0L + 2 * pi * ms
    gamma_mL = -1j * np.sqrt(kcL ** 2 - alpha_mL ** 2 + 0j)

    return ms, alpha_mL, gamma_mL

def G_spec(kcL, alpha_0L, dx_L, dz_L, num_eva):
    """naive implimentation of spectral representation"""
    adz = np.abs(dz_L)
    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, dx_L, num_eva)
    G = np.exp(-gamma_mL * adz + 1j * alpha_mL * dx_L) / gamma_mL

    return -G.sum() / 2

def G_spec_Kummar(kcL, alpha_0L, dx_L, dz_L, num_eva):
    """kummar implimentation of spectral representation"""
    adz = np.abs(dz_L)
    gamma_0L = -1j * np.sqrt(kcL ** 2 - alpha_0L ** 2 + 0j)

    ms, alpha_mL, gamma_mL = define_ms(kcL, alpha_0L, dx_L, num_eva)
    # remove m==0
    mi = ms != 0
    ms = ms[mi]
    alpha_mL = alpha_mL[mi]
    gamma_mL = gamma_mL[mi]

    u_m = np.exp(-(2 * pi * np.abs(ms) + np.sign(ms) * alpha_0L) * adz)
    u_m /= 2 * pi * np.abs(ms)
    u_m *= 1 - alpha_0L / (2 * pi * ms) + kcL ** 2 * adz / (4 * pi * np.abs(ms))

    Z = adz + 1j * dx_L
    Zc = np.conj(Z)

    S = np.exp(-alpha_0L * adz) \
      * (mp.polylog(1, np.exp(-2 * pi * Zc)) / (2 * pi)
         - (2 * alpha_0L - kcL ** 2 * adz) * mp.polylog(2, np.exp(-2 * pi * Zc)) \
            / (8 * pi ** 2))
    S += np.exp(alpha_0L * adz) \
      * (mp.polylog(1, np.exp(-2 * pi * Z)) / (2 * pi)
         + (2 * alpha_0L + kcL ** 2 * adz) * mp.polylog(2, np.exp(-2 * pi * Z)) \
            / (8 * pi ** 2))

    G = (np.exp(-gamma_mL * adz) / gamma_mL - u_m) * np.exp(2j * pi * ms * dx_L)
    G = np.exp(-gamma_0L * adz) / gamma_0L + complex(S) + G.sum()
    G *= -np.exp(1j * alpha_0L * dx_L) / 2
    return G
