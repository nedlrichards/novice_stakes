import numpy as np
from math import pi
import scipy.signal as sig

def nuttall_pulse(fc, fs):
    """Q = 1 pulse"""
    T = 1 / fc
    num_cycles = 5
    total_time = num_cycles * T
    num_samples = np.ceil(total_time * fs)
    t_signal = np.arange(num_samples) / fs
    y_signal = np.sin(2 * np.pi * fc * t_signal) *\
        sig.windows.nuttall(t_signal.size)
    return y_signal, t_signal
