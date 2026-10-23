"""
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GravityTankParams:
    """Parameters from Table 5.1(p.117).
    Water density ia taken as 62.4 lbm/ft3
    """
    g: float = 32.2            # ft/s2
    gc: float = 32.2           # lbm·ft/(lbf·s2)

    KF: float = 2.81e-2        # lbf/(ft/s)²/ft
    rho: float = 62.4          # lbm/ft3

    L: float = 3000            # ft
    Ap: float = 7.06           # ft2
    At: float = 113            # ft2

    F0: float = 35.1           # ft3/s


def f(t, y, p):
    dy0 = y[1] * p.g / p.L - p.KF * p.gc / (p.rho * p.Ap) * y[0] ** 2
    dy1 = p.F0 / p.At - p.Ap / p.At * y[0]
    dy = np.array([dy0, dy1], dtype=float)

    return dy


def steady_state(p):
    v_steady = p.F0 / p.Ap
    h_steady = p.KF * p.gc * p.L / (p.rho * p.Ap * p.g) * v_steady ** 2
    y = np.array([v_steady, h_steady], dtype=float)

    return y