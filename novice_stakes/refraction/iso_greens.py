import numpy as np
from math import pi

class IsoSpeedFan:
    """Shoot a fan of rays to the surface, create an interpolator"""

    def __init__(self, c0, z_src, num_rays, theta_min):
        """Initilize rays"""
        self.z_src = z_src
        self.c0 = c0
        self.c_src = c0
        # remove negative launch angles
        theta_min = max(0.001, theta_min)
        self.launch_angles = np.linspace(pi / 2, theta_min, num_rays)
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

        self.px = np.cos(self.launch_angles) / self.c0
        self.rho = np.abs(self.z_src) / np.tan(self.launch_angles)
        self.q = np.abs(self.z_src) / np.sin(self.launch_angles)
        self.travel_time = np.abs(self.z_src / np.sin(self.launch_angles)) / self.c0
