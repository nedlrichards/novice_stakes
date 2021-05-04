"""
============================================
Analytic cosine Fourier scatter coefficents
============================================
Reflection coefficents specific to a cosine surface.
"""
import numpy as np
from scipy.special import jv
from scipy.linalg import solve
from math import pi
from . import Bragg

class CosineRs:
    """
    Class for the computation of reflection coefficents specific to cosines
    """
    def __init__(self, H, L, c=1500, attn=0.):
        """Common parameters for cosine surface"""
        self.H = H
        self.L = L
        self.bragg = Bragg(L, c=c, attn=attn)

    def rfm_1st(self, theta_inc, qs, facous):
        """Uses the Rayleigh-Fourier method to solve for plane wave coefficents.
        Uses HIE of 1st kind to set up system of equations
        Fourier coefficents are analytic, using Bessel integral representation

        theta_inc: scalar grazing angle of incident plane wave
        facous: frequency of acoustics, 1 / s
        H: Height of cosine surface wave, m
        L: Length of cosine surface wave, m
        num_evanescent: number of evanescent plane waves to include in solution
        c: sound speed of medium, m / s

        Rs: vector of reflection coefficents
        qs: vector of the bragg order of each reflection coefficent
        """
        _, _, b0, bn = self.bragg.bragg_angles(theta_inc, qs, facous)

        # pressure release boundry condition in RFM
        b = 1j ** qs * jv(qs, -b0 * self.H / 2)
        nm_diff = qs[None, :] - qs[:, None]
        A = 1j ** nm_diff * jv(nm_diff, bn[None, :] * self.H / 2)

        # reflection coefficents that satisfy boundry condition
        rs = solve(-A, b)
        return rs

    def rfm_2nd(self, theta_inc, qs, facous):
        """Uses the Rayleigh-Fourier method to solve for plane wave coefficents.
        Uses HIE of 2nd kind to set up system of equations
        Fourier coefficents are analytic, using Bessel integral representation

        theta_inc: scalar grazing angle of incident plane wave
        facous: frequency of acoustics, 1 / s
        H: Height of cosine surface wave, m
        L: Length of cosine surface wave, m
        num_evanescent: number of evanescent plane waves to include in solution
        c: sound speed of medium, m / s

        Rs: vector of reflection coefficents
        qs: vector of the bragg order of each reflection coefficent
        """

        kacous = self.bragg.kacous(facous)

        _, an, _, bn = self.bragg.bragg_angles(theta_inc, qs, facous)

        # hka is main diagonal result for 2nd kind
        b2nd = self.ka(theta_inc, qs, facous)

        qdiff = qs[None, :] - qs[:, None]
        bdiff = bn[None, :] - bn[:, None]

        A2nd = 1j ** qdiff * (kacous ** 2 - an[None, :] * an[:, None]
             - bn[None, :] * bn[:, None]) \
             * jv(qdiff, -self.H * bdiff / 2) / (bn[:, None] * bdiff)

        A2nd = np.identity(qs.size) - np.tril(A2nd, k=-1) - np.triu(A2nd, k=1)
        rs = solve(A2nd, b2nd)
        return rs

    def ka(self, theta_inc, qs, facous):
        """
        Return a vector of the q-th reflection coefficents, HK approximation
        """
        a0, aq, b0, bq = self.bragg.bragg_angles(theta_inc, qs, facous)
        rs = 1j ** qs * jv(qs, -self.H * (b0 + bq) / 2) \
           * (a0 * qs * self.bragg.Kper / (bq * (b0 + bq)) - b0 / bq)
        return rs
