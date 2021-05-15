import numpy as np
import numexpr as ne
from math import pi

def p_sca(dpdn_g_as, g_ra, dx, sig_FT, faxis, tau_total, tau_reference,
          max_duration, spreading=None):
    """
    General framework for integrating limited duration scatter timeseries
    """
    if len(dpdn_g_as.shape) == 3:
        is_2D = True
    elif len(dpdn_g_as.shape) == 2:
        is_2D = False
        spreading = np.array(spreading, ndmin=1)
    else:
        raise(ValueError('Number of dimensions of dpdn_g_as is not 2 or 3'))


    # reference for travel time curve
    tau_i = (tau_total - tau_reference) < max_duration
    phase_shift_ = np.exp(2j * pi * faxis * tau_reference)[:, None]

    # setup terms for integral computation
    dpdn_g_ = dpdn_g_as[:, tau_i]
    g_ = g_ra[:, tau_i]
    s_ = sig_FT[:, None]
    s_term_ = 1.

    # determine scaling based on dimension number and spreading
    if is_2D:
        igrand_scale = dx ** 2
    elif spreading.size == dpdn_g_as.shape[-1]:
        # point source, calculated with stationary phase at y=0
        igrand_scale = dx
        denom = faxis[:, None] * spreading[None, tau_i]
        denom[faxis==0, :] = 1.
        s_term_ = np.sqrt(1 / denom) * np.exp(-1j * pi / 4)
        s_term_[faxis==0, :] = 0. + 0.j

    elif spreading.size == 1:
        # line source
        # TODO: clairify why the negative sign is necassary to match image
        # solution, see mathematica nb example
        igrand_scale = -dx
    else:
        raise(ValueError('Spreading specification can be either point or line'))

    # spatial integrating and time domain truncation
    igrand = ne.evaluate('s_ * phase_shift_ * s_term_ * dpdn_g_ * g_')
    p_sca_FT = igrand_scale * np.nansum(igrand, axis=-1)
    p_sca = np.fft.irfft(p_sca_FT)

    # return t-axis
    df = (faxis[-1] - faxis[0]) / (faxis.size - 1)
    taxis = np.arange(p_sca.size) / (2 * (faxis.size - 1) * df) + tau_reference

    return p_sca, taxis
