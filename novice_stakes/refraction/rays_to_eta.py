import numpy as np
from math import pi
from scipy.interpolate import interp1d, UnivariateSpline

def rays_to_surface(ray_fan, axes, eta, eta_p=None, kc=None, shadow=False):
    """extrapolate from rays at z=0 to rays at z=eta"""

    axes = np.asarray(axes)

    if np.ndim(axes) == 3:
        rho = np.linalg.norm(axes, axis=0)
        phi = np.arctan2(axes[1], axes[0])
    elif np.ndim(axes) == 2 or np.ndim(axes) > 3:
        raise(ValueError('axes ndmin must be 1 or 3 (axis_num, x, y)'))
    else:
        rho = np.abs(axes)

    if eta_p is not None:
        eta_p = np.asarray(eta_p)
        if np.ndim(axes) == 3:
            n = np.array([-eta_p[0], -eta_p[1], np.ones_like(eta_p[0])])
        elif np.ndim(axes) == 2 or np.ndim(axes) > 3:
            raise(ValueError('eta ndmins must be 1 or 3 (axis_num, x, y)'))
        else:
            n = np.array([-eta_p, np.ones_like(eta_p)])

    # relate surface position to incident angle
    px_ier = interp1d(ray_fan.rho, ray_fan.px, kind=3,
                      bounds_error=False, fill_value=np.nan)
    px_n = px_ier(rho)
    cos_n = px_n * ray_fan.c0
    sin_n = np.sqrt(1 - cos_n ** 2)
    d_rho = -eta * cos_n / sin_n

    props = np.array([ray_fan.travel_time, ray_fan.q])
    ray_ier = interp1d(ray_fan.rho, props, kind=3,
                       bounds_error=False, fill_value=np.nan)
    rays = ray_ier(rho + d_rho)

    # adjust travel time and amplitude for extra distance
    r_surf = np.sqrt(eta ** 2 + d_rho ** 2)
    travel_time = rays[0] + r_surf / ray_fan.c0

    q = rays[1] + ray_fan.c0 * r_surf / ray_fan.c_src

    if np.ndim(axes) == 1 and kc is not None:
        # line source dynamic ray amplitude
        amp = np.sqrt(np.abs(ray_fan.c0 / (ray_fan.c_src * q))) + 0j
        amp *= np.exp(3j * pi / 4) / np.sqrt(8 * pi * kc)
    else:
        # point source dynamic ray amplitude, COA (3.65)
        amp = np.sqrt(np.abs(px_n * ray_fan.c0 / (rho * q)))
        amp /= 4 * pi

    # compute ray normal derivative projection vector
    if eta_p is not None:
        cos_theta = px_n * ray_fan.c0
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        if np.ndim(eta_p) == 3:
            proj_vec = np.array([np.cos(phi) * cos_theta,
                                 np.sin(phi) * cos_theta,
                                 sin_theta])
            proj_str = 'ijk,ijk->jk'
        elif np.ndim(eta_p) == 2:
            raise(ValueError('eta_p can only have 3 or 1 dimensions'))
        else:
            proj_vec = np.array([cos_theta, sin_theta])
            proj_str = 'ij,ij->j'

        mag_proj = np.einsum(proj_str, proj_vec, n)
        amp *= mag_proj

    # shadow correction possible for 1-D surfaces
    if shadow and np.ndim(axes) == 1:
        shadow_i = _shadow(rho, d_rho)
        amp[shadow_i] = 0.

    if kc is None:
        #use chain rule to estimate second derivative of tau wrt y
        if np.ndim(axes) == 1:
            tt_ier = UnivariateSpline(ray_fan.rho,
                                      ray_fan.travel_time,
                                      k=3, s=0)
            d_tau_d_rho = tt_ier.derivative()
            d2d = d_tau_d_rho(rho) / rho
            return amp, travel_time, d2d

    return amp, travel_time

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
