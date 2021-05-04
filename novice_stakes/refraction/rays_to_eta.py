import numpy as np
import numexpr as ne
from math import pi
from scipy.interpolate import UnivariateSpline

def greens_FS_fan(ray_fan, rho, eta, faxis,
                  phi=None, eta_p=None, dz_iso=0, isline=False, shadow=False):
    """Compute scatter pressure with the KA, using ray fans"""

    c_surf = ray_fan.c0

    # 2-D calculations
    if rho.ndim == 2:
        # broadcast specifications
        xi = (np.newaxis, slice(rho.shape[0]), slice(rho.shape[1]))
        oi = (slice(faxis.size), np.newaxis, np.newaxis)
    # 1-D calculation
    else:
        # broadcast specifications
        xi = (np.newaxis, slice(rho.size))
        oi = (slice(faxis.size), np.newaxis)

    if not isline:
        if phi is None:
            tt, amp, d2d = extrapolate_fan(ray_fan, rho, eta, dz_iso,
                                        return_d2d=True, eta_p=eta_p)
        else:
            tt, amp = extrapolate_fan(ray_fan, rho, eta, dz_iso,
                                      phi=phi, eta_p=eta_p)
            d2d = None

    else:
        kaxis = 2 * pi * faxis / c_surf
        tt, amp = extrapolate_fan(ray_fan, rho, eta, dz_iso,
                                  kaxis=kaxis, return_d2d=False,
                                  eta_p=eta_p, phi=phi)
        d2d = None

    # shadow correction possible for 1-D surfaces
    if shadow and np.ndim(axes) == 1:
        shadow_i = _shadow(rho, d_rho)
        if not isline:
            amp[shadow_i] = 0.
        else:
            amp[:, shadow_i] = 0.

    # setup array broadcasting
    omega_ = 2 * pi * faxis[oi]
    if not isline:
        amp_ = amp[xi]
    else:
        amp_= amp

    tt_ = tt[xi]

    # greens function from source
    if eta_p is None:
        g_str = 'amp_ * exp(-1j * omega_ * tt_)'
    else:
        g_str = '-1j * omega_ * amp_ * exp(-1j * omega_ * tt_) / c_surf'
    greens = ne.evaluate(g_str)

    return greens, tt, d2d

def extrapolate_fan(ray_fan, rho, eta, dz_iso,
                    kaxis=None, return_d2d=False, eta_p=None, phi=None):
    """
    Extrapolate a ray from the start of the iso-speed layer to the surface
    """
    # move from z=0 to eta
    # relate surface position to incident angle
    px_ier = UnivariateSpline(ray_fan.rho, ray_fan.px, k=3, ext=2, s=0)
    eta_ = eta + dz_iso

    # iterate untill rays intersect the surface at rho
    rho_at_dz = rho.copy()

    def ray_to_surf(rho_dz):
        """extrapolate to surface using plane wave assumption"""
        px_0 = px_ier(rho_dz)
        cos_0 = px_0 * ray_fan.c0
        sin_0 = np.sqrt(1 - cos_0 ** 2)
        # total distance in iso-speed layer
        d_r = eta_ / sin_0
        # horizontal distance in iso-speed layer
        d_rho = d_r * cos_0
        return px_0, d_rho, d_r

    # One iteration of newton's method to estimate path through the iso-speed
    # layer
    if np.abs(dz_iso) > 1e-3:
        _, d_rho0, _ = ray_to_surf(rho)
        _, d_rho1, _ = ray_to_surf(rho - d_rho0)
        dd_rho = (d_rho1 - d_rho0) / d_rho0
        rho_dz = rho / (1 - dd_rho)
        px, d_rho, d_r = ray_to_surf(rho_dz)
    else:
        rho_dz = rho
        px = px_ier(rho)
        d_rho = 0
        d_r = 0

    tt_ier = UnivariateSpline(ray_fan.rho, ray_fan.travel_time, k=3, ext=2, s=0)
    q_ier = UnivariateSpline(ray_fan.rho, ray_fan.q, k=3, ext=2, s=0)

    # adjust travel time and amplitude for extra distance
    travel_time = tt_ier(rho_dz) + d_r / ray_fan.c0

    # compute amplitude
    # dynamic ray tracing variable q from COA ch. 3
    q = q_ier(rho_dz) + ray_fan.c0 * d_r / ray_fan.c_src
    if kaxis is not None:
        # line source dynamic ray amplitude
        amp = np.sqrt(np.abs(ray_fan.c0 / (ray_fan.c_src * q))) \
            * np.exp(3j * pi / 4) / np.sqrt(8 * pi * kaxis[:, None])
        amp[kaxis == 0, :] = 0. + 0.j
    else:
        # point source dynamic ray amplitude, COA (3.65)
        amp = np.sqrt(np.abs(px * ray_fan.c0 / (rho * q)))
        amp /= 4 * pi

    # compute ray normal derivative projection vector
    if eta_p is not None:
        # compute surface normal vector
        # 2D surface requires first dimension of eta_p array to be dx, dy
        if phi is not None:
            n = np.array([-eta_p[0], -eta_p[1], np.ones_like(eta_p[0])])
        else:
            n = np.array([-eta_p, np.ones_like(eta_p)])

        cos_theta = px * ray_fan.c0
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        if phi is not None:
            proj_vec = np.array([np.cos(phi) * cos_theta,
                                 np.sin(phi) * cos_theta,
                                 sin_theta])
            proj_str = 'ijk,ijk->jk'
        else:
            proj_vec = np.array([cos_theta, sin_theta])
            proj_str = 'ij,ij->j'

        mag_proj = np.einsum(proj_str, proj_vec, n)
        amp *= mag_proj

    if np.any(np.isnan(amp)):
        raise(ValueError('nans detected in amplitude output'))

    if return_d2d:
        d_tau_d_rho = tt_ier.derivative()
        # TODO: adjust for spreading in iso-speed layer
        #d2d_spread = -d_rho / d_r
        #d2d_spread[np.isnan(d2d_spread)] = 0
        d2d_spread = 0

        d2d = (d_tau_d_rho(rho_dz) + d2d_spread) / rho
        return travel_time, amp, d2d
    else:
        return travel_time, amp

def _shadow(rho, d_rho):
    """Remove ray shadows by require monotonic rho"""
    # check for an axis where rho moves past x_src =0
    grad_sign = np.sign(np.diff(rho))
    grad_sign = np.hstack([grad_sign[0], grad_sign])
    neg_grad_i = grad_sign < 0
    pos_grad_i = grad_sign >= 0
    neg_grad_start = np.argmax(neg_grad_i)
    pos_grad_start = np.argmax(pos_grad_i)

    # return non-shadowed points expecting a monotonic decrease in rho
    if np.any(neg_grad_i):
        neg_values = (rho + d_rho)[neg_grad_i]
        mono_values = np.maximum.accumulate(neg_values[::-1])
        _, no_shad_neg = np.unique(mono_values, return_index=True)
        #undo index flip
        no_shad_neg = neg_values.size - 1 - no_shad_neg
        no_shad_neg += neg_grad_start
    if np.any(pos_grad_i):
        mono_values = np.maximum.accumulate((rho + d_rho)[pos_grad_i])
        _, no_shad_pos = np.unique(mono_values, return_index=True)
        no_shad_pos += pos_grad_start

    if np.any(neg_grad_i) and np.any(pos_grad_i):
        all_i = np.hstack([no_shad_pos, no_shad_neg])
    elif np.any(neg_grad_i):
        all_i = no_shad_neg
    else:
        all_i = no_shad_pos
    shad_ind = np.ones(rho.size, dtype=np.bool)
    shad_ind[all_i] = False
    return shad_ind
