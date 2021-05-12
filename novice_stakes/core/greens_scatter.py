from time import time
import numpy as np
import numexpr as ne
from math import pi, e
import warnings

from scipy.optimize import brentq
from scipy.linalg import solve, solve_triangular
from scipy.special import hankel1, j1, y1
from scipy.signal import argrelextrema

class GreensScatter:
    """Compute surface scatter solutions"""
    def __init__(self, xaxis, zwave, zp_wave, zpp_wave, zsrc, c=1500, attn=0):
        """common surface setup"""
        self.xaxis = xaxis
        self.DX = (xaxis[-1] - xaxis[0]) / (xaxis.size - 1)
        self.zwave = zwave
        self.zp_wave = zp_wave
        self.zpp_wave = zpp_wave

        self.zsrc = zsrc

        self.c = c - 1j * attn
        self.attn = attn

        # source to surface geometry
        self.d_es = np.sqrt(self.xaxis ** 2
                           + (self.zwave - self.zsrc) ** 2)
        self.cos_es = ((self.zwave - self.zsrc)
                       - self.zp_wave * self.xaxis) / self.d_es
        self.grad_h = np.sqrt(1 + zp_wave ** 2)

        # surface to surface interactions
        self.d_ae = np.sqrt((xaxis[:, None] - xaxis) ** 2
                            + (zwave[:, None] - zwave) ** 2)
        # cosine calculations are undefined along diagonal
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.cos_ae = (zwave[:, None] - zwave
                        - zp_wave[:, None] * (xaxis[:, None] - xaxis))\
                        / self.d_ae

    def surface_field_diem1(self, facous, isfar=True, z_ff=10):
        """Compute pressure field at the surface using diem method"""
        G = self.G1(facous, z_ff=z_ff)
        p_inc = self.p_inc(facous, isfar=isfar)
        psi_1st = solve(G * self.DX, p_inc)
        return psi_1st

    def surface_field_diem2(self, facous, isfar=True, z_ff=10,
                            mask=None, is_L=False):
        """Compute pressure field at the surface using diem method"""
        G = self.G2(facous, z_ff=z_ff, mask=mask, is_L=is_L)
        hka_psi = self.hka_psi(facous, isfar=isfar)
        G *= -2 * self.DX
        G[np.diag_indices_from(G)] += 1

        if mask is not None:
            i_hka = np.diag(mask)
            G = G[np.ix_(i_hka, i_hka)]
            hka_psi = hka_psi[i_hka]

        if is_L:
            psi_diem = solve_triangular(G, hka_psi, lower=True)
        else:
            psi_diem = solve(G, hka_psi)

        return psi_diem

    def p_inc(self, facous, isfar=True):
        """incident pressure at surface"""
        if isfar:
            p_inc = np.sqrt(self.c / (facous * self.d_es)) / (4 * pi) \
                  * np.exp(1j * (2 * pi * facous * self.d_es / self.c + pi / 4))
        else:
            p_inc = 1j / 4 * hankel1(0, 2 * pi * facous * self.d_es / self.c)
        return p_inc

    def hka_psi(self, facous, isfar=True):
        """2 times the incident value of psi at surface"""
        if isfar:
            hka_psi = np.sqrt(facous / (self.c * self.d_es)) * self.cos_es \
              * np.exp(1j * (2 * pi * facous * self.d_es / self.c + 3 * pi / 4))
        else:
            hka_psi = -1j * pi * facous * self.cos_es\
                * hankel1(1, 2 * pi * facous * self.d_es / self.c) / self.c
        return hka_psi

    def first_mask(self, xrcr, zrcr, tau_max):
        """First iterate travel time mask"""
        d_ra = np.sqrt((xrcr - self.xaxis) ** 2 + (zrcr - self.zwave) ** 2)
        d_1st = d_ra[:, None] + self.d_ae + self.d_es[None, :]
        return d_1st / self.c < tau_max

    def G1(self, facous, z_ff=10):
        """compute G matrix used in HIE, 1st kind"""
        G = np.zeros((self.xaxis.size, self.xaxis.size), dtype=np.complex_)
        kc = 2 * pi * facous / self.c
        nfi = 2 * pi * facous * self.d_ae / np.real(self.c)  < z_ff
        ffi = np.bitwise_not(nfi)
        nfi[np.diag_indices_from(nfi)] = False
        G[nfi] = -1j / 4 * hankel1(0, kc* self.d_ae[nfi])
        z = kc * self.d_ae[ffi]
        c = self.c
        G_str = '-1j / 4 * sqrt(2 / (pi * z)) * exp(1j * (z - pi / 4))'
        G[ffi] = ne.evaluate(G_str)
        g_diag = hankel1(0, kc * self.DX / (2 * e)) / (4j)
        G[np.diag_indices_from(G)] = g_diag
        return G

    def G2(self, facous, z_ff=10, mask=None, is_L=False):
        """compute G matrix used in HIE, 2nd kind"""
        G = np.zeros((self.xaxis.size, self.xaxis.size), dtype=np.complex_)
        nfi = 2 * pi * facous * self.d_ae / np.real(self.c) < z_ff
        ffi = np.bitwise_not(nfi)

        # only compute left triangle
        if is_L:
            nfi = np.tril(nfi)
            ffi = np.tril(ffi)

        # Apply travel time mask
        if mask is not None:
            nfi = np.bitwise_and(nfi, mask)
            ffi = np.bitwise_and(ffi, mask)

        nfi[np.diag_indices_from(nfi)] = False
        G[nfi] = -1j * pi * facous * self.cos_ae[nfi]\
                * hankel1(1, 2 * pi * facous * self.d_ae[nfi] / self.c) \
                / (2 * self.c)
        z = 2 * pi * facous * self.d_ae[ffi] / self.c
        cae = self.cos_ae[ffi]
        c = self.c
        G_str = '-1j * pi * facous * cae * sqrt(2 / (pi * z))' \
              + '* exp(1j * (z - 3 * pi / 4)) / (2 * c)'
        G[ffi] = ne.evaluate(G_str)
        G[np.diag_indices_from(G)] = -self.zpp_wave \
                                   / (4 * pi * self.grad_h ** 2)
        return G
