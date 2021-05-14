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
        self.px = np.cos(self.launch_angles) / self.c0
        self.rho = np.abs(self.z_src) / np.tan(self.launch_angles)
        self.q = np.abs(self.z_src) / np.sin(self.launch_angles)
        self.travel_time = np.abs(self.z_src / np.sin(self.launch_angles)) / self.c0
