import numpy as np
import numexpr as ne
from math import pi
from novice_stakes.refraction import rays_to_surface

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
        s_term_ = np.sqrt(1 / (faxis[:, None] * spreading[None, tau_i])) \
                  * np.exp(-1j * pi / 4)

    elif spreading.size == 1:
        # line source
        igrand_scale = dx
        kc = spreading[0]
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

def p_sca_fan(ray_src, ray_rcr, xaxis, x_rcr, eta, eta_p,
              tau_img, tau_lim, faxis, sig_FT, spreading):

    dx = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
    omega = 2 * pi * faxis[:, None]
    c_surf = ray_src.c0

    if spreading is None:
        # stationary phase for y-integral
        isline = False
        src_amp, src_tt, src_d2d = rays_to_surface(ray_src,
                                                   xaxis,
                                                   eta,
                                                   eta_p=eta_p)
        rcr_amp, rcr_tt, rcr_d2d = rays_to_surface(ray_rcr,
                                                   np.abs(x_rcr - xaxis),
                                                   eta)

        # greens function from source
        dpdn_g_as = -1j * omega * src_amp * np.exp(-1j * omega * src_tt) / c_surf
        # greens function to receiver
        g_ra = rcr_amp * np.exp(-1j * omega * rcr_tt)

        # compute spreading factor for stationary phase approximation
        # second derivative of (d_src + d_rcr) wrt y
        spreading = src_d2d + rcr_d2d


    elif np.array(spreading).size == 1:
        isline = True
        # line source spreading
        kc = float(spreading)
        src_amp, src_tt = rays_to_surface(ray_src,
                                        xaxis,
                                        eta,
                                        eta_p=eta_p,
                                        kc=kc)

        rcr_amp, rcr_tt = rays_to_surface(ray_rcr,
                                        np.abs(x_rcr - xaxis),
                                        eta,
                                        kc=kc)
        # greens function from source
        dpdn_g_as = -1j * omega * src_amp * np.exp(-1j * omega * src_tt) / c_surf
        # greens function to receiver
        g_ra = rcr_amp * np.exp(-1j * omega * rcr_tt)


    elif spreading.size > 1:
        # 2-D calculations
        isline = False
        yaxis = spreading.copy()
        spreading = None
        # initialize 2D axes
        axes_src = np.array(np.meshgrid(xaxis, yaxis, indexing='ij'))
        axes_rcr = np.array(np.meshgrid(np.abs(x_rcr - xaxis), yaxis, indexing='ij'))

        src_amp, src_tt = rays_to_surface(ray_src,
                                          axes_src,
                                          eta,
                                          eta_p=eta_p)

        rcr_amp, rcr_tt = rays_to_surface(ray_rcr,
                                          axes_rcr,
                                          eta)

        # greens function from source
        omega_ = 2 * pi * faxis[:, None, None]
        aas_ = src_amp[None, :, :]
        ara_ = rcr_amp[None, :, :]
        ttas_ = src_tt[None, :, :]
        ttra_ = rcr_tt[None, :, :]

        ne_str = '-1j * omega_ * aas_ * exp(-1j * omega_ * ttas_) / c_surf'
        dpdn_g_as = ne.evaluate(ne_str)

        ne_str = 'ara_ * exp(-1j * omega_ * ttra_)'
        g_ra = ne.evaluate(ne_str)

    # surface integral for pressure at receiver
    p_rcr, taxis = p_sca(2 * dpdn_g_as, g_ra, dx, sig_FT, faxis,
                         src_tt + rcr_tt, tau_img, tau_lim,
                         spreading=spreading)

    # pressure refence based on source type
    if isline:
        p_ref = np.sqrt(2 / (pi * kc * c_surf * tau_img)) / 4
    else:
        p_ref = 1 / (4 * pi * tau_img * c_surf)

    return p_rcr, taxis, p_ref
