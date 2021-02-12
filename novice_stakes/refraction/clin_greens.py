import numpy as np
from math import pi

class CLinear:
    """Shoot a fan of rays to the surface, create an interpolator"""

    def __init__(self, c0, cm, z_src, num_rays, theta_min):
        """Initilize rays"""
        self.c0 = c0
        # sound speed slope. Negative values indicate increase with depth
        self.cm = cm
        self.z_src = z_src
        self.c_src = c0 + self.z_src * cm
        self.launch_angles = np.linspace(pi / 2 - 0.001, theta_min, num_rays)
        # properties at z=0
        self.px = None  # Horizontal slowness
        self.rho = None  # cylindrical range
        self.travel_time = None  # travel time
        self.q = None  # dynamic ray tracing variable COA (3.58)
        self._rays_to_z0()

    def _rays_to_z0(self):
        """propagate rays untill they hit z=0"""
        surface_rays = np.zeros((self.launch_angles.size, 4))

        px = np.empty_like(self.launch_angles)
        rho = np.empty_like(self.launch_angles)
        travel_time = np.empty_like(self.launch_angles)
        q = np.empty_like(self.launch_angles)

        # a from COA (3.209), snells law constant
        a = np.cos(self.launch_angles) / self.c_src
        px = np.arccos(a * self.c0)  # grazing angle at the surface

        # sin of angles used frequenctly
        sin_init = dsin = np.sqrt(1 - (a * self.c_src) ** 2)
        sin_final = np.sqrt(1 - (a * self.c0) ** 2)
        # radius of circluar paths
        radius = -1 / (a * self.cm)

        # positive launch angles have a turning point
        li = self.launch_angles >= 0

        rho[li] = (sin_final[li] - sin_init[li]) * radius[li]

        travel_time[li] = np.log((1 + sin_final[li]) / (a[li] * self.c0)) \
                        - np.log((1 + sin_init[li]) / (a[li] * self.c_src))

        q[li] = sin_final[li] / a[li] ** 2 - sin_init[li] / a[li] ** 2

       # negative launch angle have one turning point
        li = self.launch_angles < 0
        rho[li] = (sin_init[li] + sin_final[li]) * radius[li]

        travel_time[li] = np.log((1 + sin_final[li]) / (a[li] * self.c0)) \
                        + np.log((1 + sin_init[li]) / (a[li] * self.c_src))

        q[li] = sin_final[li] / a[li] ** 2 + sin_init[li] / a[li] ** 2

        q /= -self.c_src * self.cm
        travel_time /= np.abs(self.cm)

        self.px = px
        self.rho = rho
        self.travel_time = travel_time
        self.q = q
