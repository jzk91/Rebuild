"""
Gravity-flow tank model (Ch5.1):
    The force balance on the outlet pipe: dv/dt = g/L * h - (KF*gc)/(rho*Ap) * v**2
    The total continuity equation for the tank: dh/dt = F0/At - Ap/At * v

State vector: y = [v, h]
    v: outlet pipe velocity, ft/s
    h: tank level, ft

Steady state:
    v_steady = F0/Ap
    h_steady = (KF*gc)/(rho*Ap) * (v_steady**2) * (L/g)
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GravityTankParams:
    """Parameters from Table 5.1.
    Water density is not in the table, it is taken as 62.4 lbm/ft3
    """
    g: float = 32.2            # ft/s2
    gc: float = 32.2           # lbm·ft/(lbf·s2)

    KF: float = 2.81e-2        # lbf/(ft/s)2/ft
    rho: float = 62.4          # lbm/ft3

    L: float = 3000.0          # ft
    Ap: float = 7.06           # ft2
    At: float = 113.0          # ft2

    F0: float = 35.1           # ft3/s


def f(t, y, p):
    """
    Parameters
    ---
    t: float, time, s. Not used the model is autonomous.
    y: shape (2,), state vector, [v (ft/s), h (ft)]
    p: GravityTankParams

    Returns
    ---
    shape (2,), [dv/dt (ft/s2), dh/dt (ft/s)], float64
    """
    v, h = y[0], y[1]
    dv = p.g / p.L * h - p.KF * p.gc / (p.rho * p.Ap) * v ** 2
    dh = p.F0 / p.At - p.Ap / p.At * v
    dy = np.array([dv, dh], dtype=float)

    return dy


def steady_state(p):
    """
    Parameters
    ---
    p: GravityTankParams

    Returns
    ---
    shape (2,), [v_steady (ft/s), h_steady (ft)], float64
    """
    v_steady = p.F0 / p.Ap
    h_steady = p.KF * p.gc * p.L / (p.rho * p.Ap * p.g) * v_steady ** 2
    y = np.array([v_steady, h_steady], dtype=float)

    return y