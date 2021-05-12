import numpy as np
import numexpr as ne
from math import pi
from .rays_to_eta import greens_FS_fan
from novice_stakes import p_sca

def p_sca_KA_fan(src_fan, rcr_fan, xaxis, x_rcr, eta, eta_p,
                tau_img, tau_lim, faxis, sig_FT, tau_start=-0.5,
                yaxis=None, kc=None, dz_iso=0, shadow=False):
    """Compute scatter pressure with the KA, using ray fans"""

    dx = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
    c_surf = src_fan.c0

    if yaxis is None:
        # setup 1_D axes
        rho_src = np.abs(xaxis)
        rho_rcr = np.abs(x_rcr - xaxis)


        phi_src = None
        phi_rcr = None

        xi = (np.newaxis, slice(xaxis.size))
        oi = (slice(faxis.size), np.newaxis)

    else:
        # setup 2-D axes
        axes_src = np.array(np.meshgrid(xaxis, yaxis, indexing='ij'))
        axes_rcr = np.array(np.meshgrid(x_rcr - xaxis, yaxis, indexing='ij'))
        rho_src = np.linalg.norm(axes_src, axis=0)
        rho_rcr = np.linalg.norm(axes_rcr, axis=0)
        phi_src = np.arctan2(axes_src[1], axes_src[0])
        phi_rcr = np.arctan2(axes_rcr[1], axes_rcr[0])

        xi = (np.newaxis, slice(xaxis.size), slice(yaxis.size))
        oi = (slice(faxis.size), np.newaxis, np.newaxis)

    if kc is not None:
        isline = True
    else:
        isline = False

    dpdn_g_as, tt_as, d2d_src = greens_FS_fan(src_fan, rho_src, eta, faxis,
                                              phi=phi_src, isline=isline,
                                              eta_p=eta_p, dz_iso=dz_iso,
                                              shadow=shadow)

    g_ra, tt_ra, d2d_rcr = greens_FS_fan(rcr_fan, rho_rcr, eta, faxis,
                                         phi=phi_rcr, isline=isline,
                                         eta_p=None, dz_iso=dz_iso, shadow=shadow)

    if d2d_src is not None:
        # compute total second derivative of delay for stationary phase
        spreading = np.abs(d2d_src) + np.abs(d2d_rcr)
    else:
        spreading = None

    # surface integral for pressure at receiver
    p_rcr, taxis = p_sca(2 * dpdn_g_as, g_ra, dx, sig_FT, faxis,
                         tt_as + tt_ra, tau_img + tau_start * 1e-3, tau_lim,
                         spreading=spreading)

    # pressure refence based on source type
    if isline:
        p_ref = np.sqrt(2 / (pi * kc * c_surf * tau_img)) / 4
    else:
        p_ref = 1 / (4 * pi * tau_img * c_surf)

    return p_rcr, taxis, p_ref
