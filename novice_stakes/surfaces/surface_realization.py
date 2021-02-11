import numpy as np
import numexpr as ne
from math import pi

class Surface:
    """"Generation of surface realizations from spectrum"""

    def __init__(self, kmax, spectrum, seed=None, xaxis=None):
        """Setup random generator used for realizations
        spectrum is a scalar, 1-D array or 2-D array
        """
        self.rng = np.random.default_rng(seed=seed)
        self.kmax = kmax
        self.spectrum = np.array(spectrum, ndmin=1)
        self.dx = 2 * pi / kmax
        self.xaxis = xaxis
        self.g = 9.81
        self.km = 370  # wavenumber at GC wave phase speed minimum


        if self.spectrum.size == 1:
            if self.xaxis is None:
                raise(ValueError('scalar spectrum requires an x-axis'))
            self.yaxis = None
            self.N = 1
            self.kx = None
            self.ky = None
            self.omega = self.ldis_deepwater(self.kmax)
            self.h_rms = np.sum(self.spectrum)

        elif self.spectrum.size == self.spectrum.shape[0]:
            self.N = (self.spectrum.shape[0] - 1) * 2
            self.kx = np.arange(self.N // 2 + 1) * kmax / self.N
            self.ky = None
            self.xaxis = np.arange(self.N) * self.dx
            self.yaxis = None
            self.omega = self.ldis_deepwater(self.kx)
            self.h_rms = np.sqrt(np.sum(self.spectrum) * kmax / self.N)
        else:
            self.N = self.spectrum.shape[1]
            self.kx = np.arange(self.N // 2 + 1) * kmax / self.N
            self.ky = (np.arange(self.N) - self.N // 2) * kmax / self.N
            self.xaxis = np.arange(self.N) * self.dx
            self.yaxis = (np.arange(self.N) - self.N // 2) * self.dx
            k = np.sqrt(self.kx[:, None] ** 2 + self.ky[None, :] ** 2)
            self.omega = self.ldis_deepwater(k)
            self.h_ms = np.sum(self.spectrum)

    def realization(self):
        """Generate a realization of the surface spectrum"""

        # trivial sinusoid case
        if self.spectrum.size == 1:
            mag = self.spectrum * np.sqrt(2)
            phase = np.exp(1j * pi * self.rng.uniform(0, 2 * pi, 1))
            return mag * phase

        samps = self.rng.normal(size=2 * self.spectrum.size,
                                scale=1 / np.sqrt(2))
        samps = samps.view(np.complex128)

        # 1-D wave field
        if self.spectrum.size == self.spectrum.shape[0]:
            # inverse scaling of Heinzel et al. (2002), Eq 24
            abs_k2 = self.spectrum * self.N * self.kmax / 2
            return samps * np.sqrt(abs_k2)

        # 2-D wave field
        samps = samps.reshape(self.spectrum.shape)
        abs_k2 = self.spectrum * (self.N * self.kmax) ** 2
        realization = np.sqrt(abs_k2) * samps
        return realization

    def surface_synthesis(self, realization, time=None, derivative=None):
        """Synthesize surface at given time"""
        if time is not None:
            omega = self.omega
            phase = "exp(-1j * omega * time)"
        else:
            phase = "1."

        if self.spectrum.size == 1:
            kmax = self.kmax
            if derivative:
                phase += " * -1j * kmax "
            surface = np.real(phase
                              * realization
                              * np.exp(-1j * self.kmax * self.xaxis))
            return surface

        # 1-D wave field
        if self.spectrum.size == self.spectrum.shape[0]:
            if derivative:
                kx = self.kx
                phase += " * -1j * kx"
            phase += " * realization"
            surface = np.fft.irfft(ne.evaluate(phase))

        # 2-D wave field
        else:
            if derivative is not None:
                if derivative == 'x':
                    kx = self.kx[:, None]
                    phase += " * -1j * kx "
                elif derivative == 'y':
                    ky = self.ky[None, :]
                    phase += " * -1j * ky"
                else:
                    raise(ValueError('Derivative must be either x or y'))
            phase += " * realization"
            spec = ne.evaluate(phase)
            surface = np.fft.irfft2(np.fft.ifftshift(spec, axes=1),
                                    axes=(1,0))

        return surface

    def ldis_deepwater(self, wave_number):
        """linear dispersion relationship assuming deep water"""
        gc = (1 + (wave_number / self.km) ** 2)
        omega = np.sqrt(self.g * wave_number * gc)
        return omega
