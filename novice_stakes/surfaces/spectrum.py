import numpy as np
from math import pi

g = 9.81
km = 370  # wavenumber at GC wave phase speed minimum

def PM(axis_in, U20, is_omega=True):
    """Compute the Pierson-Moskowitz spectrum by radial frequency"""

    if is_omega:
        omega = np.array(axis_in, ndmin=1)
    else:
        wave_number = np.array(axis_in, ndmin=1)
        # use deep-water dispersion to compute frequency axis
        omega = ldis_deepwater(wave_number)
        dwdk = ldis_deepwater(wave_number, derivative=True)

    # Pierson Moskowitz parameters
    beta = 0.74
    alpha = 8.1e-3

    pm = np.exp(-beta * g ** 4 / (omega * U20) ** 4)
    pm *= alpha * g ** 2 / omega ** 5
    if not is_omega:
        pm *= dwdk

    pm[omega <= 0] = 0
    pm[np.isnan(pm)] = 0

    return pm

def directional_spectrum(delta, bearing, k_grid, omni_spectrum):
    """calculate spreading function from delta"""
    spreading = (1 + delta * np.cos(2 * bearing)) / (2 * pi)

    # multiply omni-directional spectrum with spreading function
    d_spec = omni_spectrum * spreading / k_grid

    d_spec[np.isnan(spreading)] = 0
    return d_spec

# directional spectrum formulations
def e_delta(wave_number, U20):
    """Elfouhaily Delta function"""

    # define wave speeds
    cphase = ldis_deepwater(wave_number) / wave_number
    cpeak = 1.14 * U20
    cmin = 0.23  # m/s

    # compute wind stress following Wu 1991
    U10 = U20 / 1.026
    C10 = (0.8 + 0.065 * U10) * 1e-3
    u_star = np.sqrt(C10) * U10

    # Elfouhaily paramters
    a0 = np.log(2) / 4
    ap = 4
    am = 0.13 * u_star / cmin

    delta = np.tanh(a0
                    + ap * (cphase / cpeak) ** 2.5
                    + am * (cmin / cphase) ** 2.5)
    return delta

def ldis_deepwater(wave_number, derivative=False):
    """linear dispersion relationship assuming deep water"""
    gc = (1 + (wave_number / km) ** 2)
    if derivative:
        dwdk = g * (1 + 3 * (wave_number / km) ** 2) \
                / (2 * np.sqrt(g * wave_number * gc))
        return dwdk
    else:
        omega = np.sqrt(g * wave_number * gc)
        return omega
