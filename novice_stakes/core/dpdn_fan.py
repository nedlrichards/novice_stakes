import numpy as np
import numexpr as ne
from math import pi
from novice_stakes.refraction import rays_to_surface

def dpdn_iter_fan(ray_src, axes_src, eta, eta_p,
                  tau_img, tau_lim, faxis, sig_FT, spreading, dz_iso=0,
                  shadow=False):
    """Compute scatter pressure with the iterative method"""

    c_surf = ray_src.c0
    dpdn_g_as, spreading = dpdn_KA_fan(ray_src, axes_src, axes_rcr, eta, eta_p,
                                       tau_img, tau_lim, faxis, sig_FT,
                                       spreading, dz_iso=dz_iso, shadow=shadow)

    return dpdn_g_as, spreading
