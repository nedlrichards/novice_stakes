import numpy as np
from scipy.optimize import newton
from .rays_to_eta import extrapolate_fan

def initialize_axes(src_fan, rcr_fan, tau_lim, x_rcr, dx, dz_iso=0, fudgef=5):
    """
    Use flat surface delay to initialize the x&y axes with delays past tau_lim
    """
    def tau_flat(x, y):
        """delay of single bounce ray to flat surface"""
        x = np.array(x)
        xs = np.sqrt(x ** 2 + y ** 2)
        tt_s = extrapolate_fan(src_fan, xs, np.zeros_like(xs), dz_iso)[0]
        xr = np.sqrt((x_rcr - x) ** 2 + y ** 2)
        tt_r = extrapolate_fan(rcr_fan, xr, np.zeros_like(xr), dz_iso)[0]
        return tt_s + tt_r

    # avoid sampling at the elements to avioid vertical ray
    d_off = 0.01 * x_rcr
    x_test = np.arange(np.ceil((x_rcr - 2 * d_off) / dx)) * dx + d_off
    tau_total = tau_flat(x_test, np.zeros_like(x_test))
    # find image ray delay and position at z=0
    i_img = np.argmin(tau_total)
    x_img = x_test[i_img]
    tau_img = tau_total[i_img]

    rooter = lambda x: tau_flat(x, np.zeros_like(x)) - tau_img - tau_lim
    xbounds = (newton(rooter, d_off) - fudgef,
               newton(rooter, x_rcr - d_off) + fudgef)

    numx = int(np.ceil((xbounds[1] - xbounds[0]) / dx)) + 1
    if numx % 2: numx += 1
    xaxis = np.arange(numx) * dx + xbounds[0]

    # iterative process to compute yaxis
    # x_ref is best guess for x position of travel time minimum at y_max
    x_ref = x_img

    for i in range(10):
        # setup yaxis
        rooter = lambda y: tau_flat(x_ref, y) - tau_img - tau_lim
        ymax = newton(rooter, tau_lim * 1500.) + fudgef
        # compute x-postion of travel time minimum at y_max
        tau_max = tau_flat(xaxis, np.full_like(xaxis, ymax))

        x_nxt = xaxis[np.argmin(tau_max)]
        if x_ref - x_nxt == 0:
            break
        x_ref = x_nxt

    numy = int(np.ceil((2 * ymax / dx))) + 1
    if numy % 2: numy += 1
    yaxis = np.arange(numy) * dx - ymax

    return xaxis, yaxis, tau_img
