import numpy as np
from scipy.optimize import newton
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


def initialize_axes(tau_src, tau_rcr, tau_lim, x_rcr, dx, fudgef=5):
    """
    Use flat surface delay to initialize the x&y axes with delays past tau_lim
    """

    tau_flat = lambda x: tau_src(x) + tau_rcr(np.abs(x_rcr - x))

    x_test = np.arange(np.ceil(x_rcr * 1.2 / dx)) * dx
    tau_total = tau_flat(x_test)
    # find image ray delay and position at z=0
    i_img = np.argmin(tau_total)
    x_img = x_test[i_img]
    tau_img = tau_total[i_img]

    rooter = lambda x: tau_flat(x) - tau_img - tau_lim
    xbounds = (newton(rooter, 0) - fudgef, newton(rooter, x_rcr) + fudgef)

    numx = int(np.ceil((xbounds[1] - xbounds[0]) / dx)) + 1
    if numx % 2: numx += 1
    xaxis = np.arange(numx) * dx + xbounds[0]

    # iterative process to compute yaxis
    # x_ref is best guess for x position of travel time minimum at y_max
    x_ref = x_img

    for i in range(10):
        # setup yaxis
        rho_src = lambda y: np.sqrt(x_ref ** 2 + y ** 2)
        rho_rcr = lambda y: np.sqrt((x_rcr - x_ref) ** 2 + y ** 2)
        tau_flat = lambda y: tau_rcr(rho_rcr(y)) + tau_src(rho_src(y))
        rooter = lambda y: tau_flat(y) - tau_img - tau_lim
        ymax = newton(rooter, tau_lim * 1500.) + fudgef
        # compute x-postion of travel time minimum at y_max
        tau_max = tau_src(np.sqrt(xaxis ** 2 + ymax ** 2)) \
                + tau_rcr(np.sqrt((x_rcr - xaxis) ** 2 + ymax ** 2))

        x_nxt = xaxis[np.argmin(tau_max)]
        if x_ref - x_nxt == 0:
            break
        x_ref = x_nxt

    numy = int(np.ceil((2 * ymax / dx))) + 1
    if numy % 2: numy += 1
    yaxis = np.arange(numy) * dx - ymax

    return xaxis, yaxis, tau_img
