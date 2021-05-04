"""
===========================================
Plane wave reflection from periodic surface
===========================================
Common framework for scatter with a plane wave source and a periodic surface
"""

import numpy as np
from math import pi

class Bragg:
    """Compute reflection coefficents for cosine surface"""
    def __init__(self, Lper, c=1500., attn=0):
        """surface specific properties
        attn: volume attenuation, dB / km"""
        self.Lper = Lper
        self.Kper = 2 * pi / Lper
        self.c = c
        self.attn = attn
        self.delta = lambda k: 1j * self.attn / 8686.\
                               / np.real(k + np.spacing(1))

    def kacous(self, facous):
        """complex wavenumeber of medium"""
        k = 2 * pi * facous / self.c
        return k + 1j * self.delta(k)

    def xsampling(self, facous, decimation=8):
        """Make a periodically sampled xaxis"""
        dx = self.c / (decimation * facous)
        numx = int(np.ceil(self.L / dx))
        dx = self.L / numx
        xaxis = np.arange(numx) * dx
        return xaxis, dx

    def qvec(self, theta_inc, num_eva, facous):
        """Return vector of bragg grating orders
        cutoff after num_eva evanescent orders on each side
        """
        kacous = np.real(self.kacous(facous))
        kx = np.real(np.cos(theta_inc) * kacous)
        num_p = np.fix((kacous - kx) / self.Kper) + num_eva
        num_n = np.fix((kacous + kx) / self.Kper) + num_eva
        qvec = np.arange(-num_n, num_p + 1)
        return qvec

    def bragg_angles(self, theta_inc, qs, facous):
        """Computer the brag angle cosine vectors"""
        kacous = self.kacous(facous)

        # compute bragg orders
        a0 = np.real(np.cos(theta_inc) * kacous)
        b0 = np.conj(np.sqrt(kacous ** 2 - a0 ** 2))
        aq = a0 + qs * self.Kper
        bq = np.conj(np.sqrt(kacous ** 2 - aq ** 2))
        return a0, aq, b0, bq

    def p_sca(self, theta_inc, qs, facous, rs, xsrc, zsrc, xrcr, zrcr):
        """
        Scattered pressure field from plane wave reflection coefficents
        """
        a0, aq, b0, bq = self.bragg_angles(theta_inc, qs, facous)
        phase = -a0 * xsrc - b0 * zsrc + aq * xrcr - bq * zrcr
        p_sca = rs @ np.exp(-1j * phase)
        return p_sca

    def r_energy(self, rs, theta_inc, qs, facous):
        """Calculate the energy conservation relative to 1"""
        kacous = self.kacous(facous)
        _, aq, b0, bq = self.bragg_angles(theta_inc, qs, facous)
        # compute energy
        reali = np.abs(np.real(aq ** 2)) <= np.real(kacous) ** 2
        en_conn = np.abs(rs[reali]) ** 2 * np.real(bq[reali]) / np.real(b0)
        return np.sum(en_conn)
