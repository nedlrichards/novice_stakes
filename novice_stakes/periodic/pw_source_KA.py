import numpy as np
import numexpr as ne
from math import pi
from novice_stakes import p_sca
from scipy.optimize import newton

def initialize_axis_pw(z_src, z_rcr, x_rcr, dx, tau_lim, c0=1500.):
    """ initialize axis assuming a plane wave source"""
    x_img = z_src * x_rcr / (z_src + z_rcr)
    theta_inc = np.arctan(np.abs(z_src) / x_img)
    tau_img = np.sqrt((z_src + z_rcr) ** 2 + x_rcr ** 2) / c0
    tau_img_src = np.sqrt(z_src ** 2 + x_img ** 2) / c0

    # setup xaxis based on maximum delay
    px = np.cos(theta_inc) / c0
    pz = np.sin(theta_inc) / c0

    rooter = lambda x: px * (x - x_img) \
                       + np.sqrt((x_rcr - x) ** 2 + z_rcr ** 2) / c0
    tau_min = rooter(x_img)


    x1 = newton(lambda x: rooter(x) - tau_min - tau_lim, 0)
    x2 = newton(lambda x: rooter(x) - tau_min - tau_lim, x_rcr)

    ff = 5
    numx = np.ceil((x2 - x1 + ff) / dx)
    xaxis = np.arange(numx) * dx - ff / 2 + x1
    return xaxis, x_img, tau_min


def p_sca_KA_iso_pw(z_src, z_rcr, x_rcr, xaxis, eta, eta_p, tau_lim, faxis, fc,
                    sig_FT, c0=1500., tau_start=-0.5, shadow=False):
    """Compute scatter pressure with the KA, using ray fans"""

    # plane wave source
    dx = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
    x_img = z_src * x_rcr / (z_src + z_rcr)
    theta_inc = np.arctan(np.abs(z_src) / x_img)
    tau_img = np.sqrt((z_src + z_rcr) ** 2 + x_rcr ** 2) / c0
    tau_img_src = np.sqrt(z_src ** 2 + x_img ** 2) / c0
    tau_img_rcr = np.sqrt(z_rcr ** 2 + (x_rcr - x_img) ** 2) / c0

    # setup xaxis based on maximum delay
    px = np.cos(theta_inc) / c0
    pz = np.sin(theta_inc) / c0

    tt_as = (xaxis - x_img) * px + eta * pz + tau_img_src
    n = np.array([-eta_p, np.ones_like(eta_p)])
    grad_g_as = 2j * pi * faxis[None, None, :] * np.array([px, pz])[:, None, None] \
              * np.exp(2j * pi * faxis[None, None, :] * tt_as[None, :, None])
    dpdn_g_as = np.einsum('ij,ijk->jk', n, grad_g_as)


    rho_rcr = np.sqrt((x_rcr - xaxis) ** 2 + (z_rcr - eta) ** 2)
    tt_ra = rho_rcr / c0
    g_ra = 1j * np.sqrt(c0 / (rho_rcr[:, None] * faxis)) \
         * np.exp(1j * (2 * pi * faxis * tt_ra[:, None] - pi / 4)) / (4 * pi)

    # surface integral for pressure at receiver
    p_rcr, taxis = p_sca(2 * np.conj(dpdn_g_as).T, np.conj(g_ra).T,
                         dx, sig_FT, faxis,
                         tt_as + tt_ra, tau_img + tau_start * 1e-3,
                         tau_lim)

    return p_rcr, taxis
