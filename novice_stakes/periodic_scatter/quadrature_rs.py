"""
===================================================
Solve for with Fourier coefficents of scatter field
===================================================
Reflection coefficents for a general periodic surface.
"""
import numpy as np
from scipy.special import jv
from scipy.linalg import solve
import numexpr as ne
from math import pi
from . import Bragg

class QuadRs:
    """
    Class for the computation of reflection coefficents using DFT
    """
    def __init__(self, xaxis, zwave, zp_wave, c=1500, attn=0.):
        """Common parameters for cosine surface"""
        self.xaxis = xaxis
        self.zwave = zwave
        self.zp_wave = zp_wave
        self.DX = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
        self.L = self.DX * self.xaxis.size
        self.bragg = Bragg(self.L, c, attn)

    def ka(self, theta_inc, qs, facous):
        """
        reflection coefficents are calculated from incident field

        theta_inc: scalar grazing angle of incident plane wave
        facous: frequency of acoustics, 1 / s
        num_evanescent: number of evanescent plane waves to include in solution

        rs: vector of reflection coefficents
        """
        a0, _, b0, _ = self.bragg.bragg_angles(theta_inc, qs, facous)
        kacous = self.bragg.kacous(facous)

        # normal derivative of pressure at surface
        projection = np.dot(np.array([a0, b0]),
                            np.array([-self.zp_wave, np.ones_like(self.xaxis)]))
        dpinc = -2j * projection * np.exp(-1j * b0 * self.zwave)

        # integrate surface integral for reflection coefficents
        rs = self._r_from_dpdn(dpinc, theta_inc, qs, facous)
        return rs

    def rfm_1st(self, theta_inc, qs, facous):
        """
        Uses the Rayleigh-Fourier method to solve for plane wave coefficents.
        The HIE of the 1st kind is used to derive system of equations
        Fourier coefficents are calculated using rectangular quadrature

        theta_inc: scalar grazing angle of incident plane wave
        facous: frequency of acoustics, 1 / s
        num_evanescent: number of evanescent plane waves to include in solution

        rs: vector of reflection coefficents
        """
        _, _, b0, bn = self.bragg.bragg_angles(theta_inc, qs, facous)

        # make sure there are enough orders in calculation to drop Nyquest
        while self.xaxis.size // 2 <= np.max(np.abs(qs)):
            raise(ValueError('Increase sampling of surface'))

        # compute scattering orders, note this is not standard order!
        ncompute = np.arange(self.xaxis.size) - (self.xaxis.size + 1) // 2
        nstarti = int(np.where(ncompute == qs[0])[0])
        nendi = int(np.where(ncompute == qs[-1])[0])

        # compute Fourier series of incident pressure field
        pinc = -np.exp(1j * b0 * self.zwave)
        pinc_FT = np.fft.fftshift(np.fft.fft(pinc))
        pinc_FT /= self.xaxis.size

        # scattered pressure field with unit amplitude reflection coefficents
        phase = qs[None, :] * self.bragg.Kper * self.xaxis[:, None] \
              - bn * self.zwave[:, None]
        psca = np.exp(1j * phase)

        psca_FT = np.fft.fftshift(np.fft.fft(psca, axis=0), axes=0)
        psca_FT /= self.xaxis.size

        # remove high order evanescent waves before inverse computation
        pinc_FT = pinc_FT[nstarti: nendi + 1]
        psca_FT = psca_FT[nstarti: nendi + 1, :]

        rs = solve(psca_FT, pinc_FT)
        return rs

    def psi_hie_1st(self, theta_inc, qs, facous):
        """
        Uses a helmholtz integral equation (HIE) of the first kind to compute
        normal derivative of the pressure field at the surface.

        theta_inc: scalar grazing angle of incident plane wave
        facous: frequency of acoustics, 1 / s
        num_n: accelarated sum for HIE includes 2N+1 terms per matrix entry

        phi: normal derivative of pressure field at the surface. Result is not
        scaled by amplitude of surface normal vector.
        """
        L = self.L
        Kper = self.Kper
        a0, _, b0, bn = self.bragg.bragg_angles(theta_inc, qs, facous)
        kacous = self.bragg.kacous(facous)

        # compute incident pressure field
        p_inc = np.exp(1j * b0 * self.zwave)

        # compute position differnces
        dx = self.xaxis[:, None] - self.xaxis[None, :]
        dz = np.abs(self.zwave[:, None] - self.zwave[None, :])

        # compute main diagonal contribution
        # difference between G and Ginfty
        gd = np.sum(1j / (2 * L * bn)) \
           - np.sum(1 / (4 * pi * np.abs(qs[qs != 0])))
        # Ginfty continous contribution
        gip_cont = np.log(2 * pi) + np.log(1 + self.zp_wave ** 2) / 2
        gip_cont /= -(2 * pi)
        # Ginfty singularity contribution
        gip_sing = (np.log(np.abs(dx[:, 0]) / L) \
                 + np.log(1 - np.abs(dx[:, 0]) / L))
        gip_sing /= (2 * pi)
        gip_sing = np.sum(gip_sing[1: ]) * self.DX + L / pi
        # add all contributions for main diagonal
        gdiag = (gd + gip_cont) * self.DX + gip_sing

        # limit the size of each matrix
        nq = 10

        #limit qs to a multiple of nq
        qrem = (qs.size - 1) % nq

        qi = np.zeros(qs.shape, dtype=np.bool)
        if qrem == 0:
            qi[:] = 1
        else:
            qi[qrem // 2: -qrem // 2] = 1
        qs = qs[qi]
        bn = bn[qi]
        ier = np.split(np.arange(qs.size - 1), nq, axis=-1)

        # compute naive series term
        xs = dx[:, :, None]
        zs = dz[:, :, None]
        # treat q zero term as a special case
        kzns = bn[None, None, qs != 0]
        qnes = qs[None, None, qs != 0]

        # use qx / abs(qx) as a sign function
        nes = """1j * exp(1j * (bx * zs + qx * Kper * xs)) / (2 * L * bx) \
                 - exp(-qx * a0 * zs / abs(qx)) \
                 * exp(qx * Kper * (1j * xs - qx * zs / abs(qx))) \
                 / (4 * pi * abs(qx))"""

        # chunk calculation to be gentle on memory
        num_chunks = int(np.ceil(qs.size / nq))

        # start Gf with zero-th term
        Gf = 1j * np.exp(1j * b0 * dz) / (2 * L * b0)

        # Add non zero terms
        for ix in ier:
            bx = kzns[:, :, ix]
            qx = qnes[:, :, ix]
            temp = ne.evaluate(nes)
            temp = np.sum(temp, axis=-1)
            Gf += temp

        # compute positive asymptotic sum result
        Gp_total = np.exp(-a0 * dz) * np.log(1 - np.exp(Kper * (1j * dx - dz)))
        Gp_total /= -4 * pi
        Gn_total = np.exp(a0 * dz) * np.log(1 - np.exp(-Kper * (1j * dx + dz)))
        Gn_total /= -4 * pi

        # sum results together to approximate final matrix
        Gf += Gp_total + Gn_total
        Gf *= self.DX

        # solve for psi_total
        Gf[np.diag_indices_from(Gf)] = gdiag
        psi_1st = solve(Gf, -p_inc)

        return psi_1st

    def _r_from_dpdn(self, dpdn, theta_inc, qvec, facous):
        """
        reflection coefficents calculated from p normal derivative at surface
        """
        _, _, _, bn = self.bragg.bragg_angles(theta_inc, qvec, facous)
        # greens function at surface
        phase = (bn[:, None] * self.zwave[None, :]
                - qvec[:, None] * self.bragg.Kper * self.xaxis[None, :])
        gra = (1j / (2 * self.L)) * np.exp(-1j * phase) / bn[:, None]

        # integrate surface integral for reflection coefficents
        rs = -np.sum(dpdn * gra, axis=1) * self.DX
        return rs
