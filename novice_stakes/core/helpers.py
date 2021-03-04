import numpy as np
from .pulse_signal import nuttall_pulse

def initialize_nuttall(fc, fs, c_surf, tau_lim, decimation=8, num_dither=5):
    """
    initialize time and frequency sampling consistant with tau_lim and xmission
    """

    dx = c_surf / (decimation * fc)

    # compute time/frequency domain parameters

    # transmitted signal
    sig_y, sig_t = nuttall_pulse(fc, fs)

    # compute t and f axes
    num_t = int(np.ceil(tau_lim * fs + sig_y.size + num_dither))
    if num_t % 2: num_t += 1

    # flat surface specifications
    # compute FT of transmitted signal
    faxis = np.arange(num_t // 2 + 1) * fs / num_t
    sig_FT = np.fft.rfft(sig_y, num_t)

    return faxis, dx, sig_FT
