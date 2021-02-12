import numpy as np
from scipy.interpolate import interp1d

def rays_to_surface(ray_fan, axes, eta, eta_p=None):
    """extrapolate from rays at z=0 to rays at z=eta"""

    rho = np.linalg.norm(axes, axis=0)


    # relate surface position to launch angle
    la_ier = interp1d(ray_fan.rho, ray_fan.launch_angle)
    la_n = la_ier(rho)
    return la_n


