import numpy as np
from math import pi
from scipy.interpolate import interp1d, UnivariateSpline

def rays_to_surface(ray_fan, axes, eta, c_src, eta_p=None, c_surf=1500., return_d2tau=False):
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



    # relate surface position to launch angle
    la_ier = interp1d(ray_fan.rho, ray_fan.launch_angles, kind=3,
                      bounds_error=False, fill_value=np.nan)
    la_n = la_ier(rho)

    # TODO: Add a newton iteration or two here
    la_n1 = la_ier(rho)

    # interpolate ray properties on the surface
    props = np.array([ray_fan.px, ray_fan.travel_time, ray_fan.q])
    ray_ier = interp1d(ray_fan.launch_angles, props, kind=3,
                      bounds_error=False, fill_value=np.nan)
    rays = ray_ier(la_n1)

    travel_time = rays[1]
    # dynamic ray amplitude from COA (3.65)
    amp = np.sqrt(np.abs(rays[0] * c_src / (rho * rays[2]))) / (4 * pi)

    # compute ray normal derivative projection vector
    if eta_p is not None:
        cos_theta = rays[0] * c_surf
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        if np.ndim(eta_p) == 3:
            proj_vec = np.array([np.cos(phi) * cos_theta,
                                 np.sin(phi) * cos_theta,
                                 sin_theta])
            proj_str = 'ijk,ijk->jk'
        else:
            proj_vec = np.array([cos_theta, sin_theta])
            proj_str = 'ij,ij->j'

        mag_proj = np.einsum(proj_str, proj_vec, n)

    if return_d2tau:
        #use chain rule to estimate second derivative of tau wrt y
        if np.ndim(axes) != 1:
            raise(ValueError("Stationary phase assumes 1D surface"))
        tt_ier = UnivariateSpline(ray_fan.rho, ray_fan.travel_time, k=3, s=0)
        d_tau_d_rho = tt_ier.derivative()

        d2d = d_tau_d_rho(rho) / rho

        return amp, travel_time, d2d

    return amp, travel_time
