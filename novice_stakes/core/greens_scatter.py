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

        # put off surface to surface interactions untill needed
        self.d_ae = None
        self.cos_ae = None

    def surface_field_hka(self, facous, isfar=True):
        """Expression for the unshadowed HKA"""
        hka_psi = self.hka_psi(facous, isfar=isfar)
        return hka_psi

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

    def eigs_at_rcr(self, facous, xrcr, zrcr):
        """compute delay and amplitude of eigen-rays to receiver"""
        k_acu = 2 * pi * facous / self.c
        d_ra = np.sqrt((xrcr - self.xaxis) ** 2 + (zrcr - self.zwave) ** 2)
        d = (self.d_es + d_ra)
        # analytic differentiation from mathematica
        d_pp = -((xrcr - self.xaxis) + (zrcr - self.zwave) * self.zp_wave) ** 2 \
               / ((zrcr - self.zwave) ** 2 + (xrcr - self.xaxis) ** 2) ** (3 / 2) \
               -(self.xaxis + (self.zwave - self.zsrc) * self.zp_wave) ** 2 \
               / ((self.zwave - self.zsrc) ** 2 + self.xaxis ** 2) ** (3 / 2) \
               + (1 + self.zp_wave ** 2 - (zrcr - self.zwave) * self.zpp_wave) \
               / np.sqrt((zrcr - self.zwave) ** 2 + (xrcr - self.xaxis) ** 2) \
               + (1 + self.zp_wave ** 2 + (self.zwave - self.zsrc) * self.zpp_wave) \
               / np.sqrt((self.zwave - self.zsrc) ** 2 + self.xaxis ** 2)

        # find local min and max on tau
        mins = argrelextrema(d, np.greater)[0]
        maxs = argrelextrema(d, np.less)[0]
        all_eigs = np.hstack([mins, maxs])

        amp = -self.cos_es / (4 * pi * np.sqrt(d_ra * self.d_es))
        sta_res = amp * np.sqrt(2 * pi / (k_acu * np.abs(d_pp))) \
                * np.exp(1j * (k_acu * d + np.sign(d_pp) * pi / 4))
        return self.xaxis[all_eigs], sta_res[all_eigs]

    def hka_psi(self, facous, isfar=True):
        """2 times the incident value of psi at surface"""
        if isfar:
            hka_psi = np.sqrt(facous / (self.c * self.d_es)) * self.cos_es \
              * np.exp(1j * (2 * pi * facous * self.d_es / self.c + 3 * pi / 4))
        else:
            hka_psi = -1j * pi * facous * self.cos_es\
                * hankel1(1, 2 * pi * facous * self.d_es / self.c) / self.c
        return hka_psi

    def grcr(self, xrcr, zrcr, facous):
        """compute vector to source and receiver from surface"""
        # surface to receiver geometry
        d_ra = np.sqrt((xrcr - self.xaxis) ** 2 + (zrcr - self.zwave) ** 2)
        grcr = np.sqrt(self.c / (facous * d_ra)) / (4 * pi) \
                * np.exp(1j * (2 * pi * facous * d_ra / self.c + pi / 4))
        return grcr

    def first_mask(self, xrcr, zrcr, tau_max):
        """First iterate travel time mask"""
        self._compute_selfinteraction()
        d_ra = np.sqrt((xrcr - self.xaxis) ** 2 + (zrcr - self.zwave) ** 2)
        d_1st = d_ra[:, None] + self.d_ae + self.d_es[None, :]
        return d_1st / self.c < tau_max

    def G1(self, facous, z_ff=10):
        """compute G matrix used in HIE, 1st kind"""
        self._compute_selfinteraction()
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
        self._compute_selfinteraction()

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

    def shadow_test(self, xrcr, zrcr, issrc=True):
        """compute geometric shadow zones on surface
        """
        if issrc:
            local_x = self.xaxis
            la = np.arctan2(local_x, self.zwave - self.zsrc)
        else:
            local_x = (xrcr - self.xaxis)[: : -1]
            la = np.arctan2(local_x, (self.zwave - zrcr)[: : -1])

        # only consider forward launch angles
        starti = local_x > 0
        if np.any(starti):
            starti = np.argmax(starti)
        else:
            starti = 0

        la = la[starti: ]
        # shadow zones start when difference goes to zero
        la_diff = np.diff(la)
        sstart = np.sign(la_diff)
        ssi = np.where(sstart == -1)[0]
        szone = []
        istart = 0
        while istart < ssi.size:
            shadow_la = la[ssi[istart]]
            exceedence = la[ssi[istart]: ] > shadow_la
            if not np.any(exceedence):
                szone.append([ssi[istart] + starti, self.xaxis.size - 1])
                break
            endi = np.argmax(exceedence)
            szone.append([ssi[istart] + starti, ssi[istart] + endi + starti])

            # move to next shadow zone
            istart = np.where(ssi > ssi[istart] + endi)[0]
            if istart.size == 0:
                break
            istart = istart[0]
        # flip shadow points if necassary for receiver local coordinates
        if not issrc:
            nx = local_x.size - 1
            szone = [[nx - sz[1], nx - sz[0]] for sz in szone]
        return szone

    def _compute_selfinteraction(self):
        """Only compute self interaction matricies for HIE calculations"""
        xaxis = self.xaxis
        zwave = self.zwave
        zp_wave = self.zp_wave

        if self.d_ae is None:
            # surface to surface interactions
            self.d_ae = np.sqrt((xaxis[:, None] - xaxis) ** 2
                    + (zwave[:, None] - zwave) ** 2)
            # cosine calculations are undefined along diagonal
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.cos_ae = (zwave[:, None] - zwave
                            - zp_wave[:, None] * (xaxis[:, None] - xaxis))\
                            / self.d_ae
