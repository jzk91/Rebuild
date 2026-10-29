"""
Simplified binary distillation column model (Ch5.6):

Assumptions:
---
1. Equimolal overflow. (no energy equation needed)
2. Negligible vapor holdup. (vapor rates through all trays are the same)
3. Constant relative volatility.
4. Theoretical trays.
5. Total condenser.
6. Constant holdups in the reflux drum and the column base.
7. Feed is saturated liquid.

Equations: 4NT+11
---
For each tray (n = 1, ..., NT, 2NT equations)
1. total continuity for each tray: 
dM(n)/dt = L(n+1) - L(n)                                               # L(NT+1)  = R 
2. component continuity for each tray: 
dM(n)x_L(n)/dt = L(n+1)x_L(n+1) - L(n)x_L(n) + V*(y_V(n-1)-y_V(n))     # y_V(0) = yB

For feed tray (n=NF)
dM(n)/dt += F
dM(n)x(n)/dt += F*z

For reflux drum (2 equations)
dMD/dt= V - R - D
dMDxD/dt = V*y_V(NT) - R*xD - D*xD

For column base (2 equations)
dMB/dt = L(1) - B - V
dMBxB/dt = L(1)x_L(1) - V*yB - B*xB

Equilibrium (trays plus base; NT+1 equations)
y_V(n) = alpha * x_L(n) / (1 + (alpha - 1) * x_L(n))

Hydraulics (trays; NT equations)
L(n) = L0(n) + (M(n) - M0) / beta
L0(n) = R0 + F; n = 1, 2, ..., NF; saturated liquid feed goes down
L0(n) = R0; n = NF+1, ..., NT

Level controllers(2 equations)
dMD/dt = 0; reflux drum
dMB/dt = 0; column base

Proportional-integral feedback controllers(4 equations)
R = R0 + KcD·(eD + ∫eD dt / τD), 
V = V0 - KcB·(eB + ∫eB dt / τB), 
erintD = ∫eD dt, eD = xD_set - xD
erintB = ∫eB dt, eB = xB_set - xB

Variables: 4NT+11
---
x_L, y_V, M, L; shape(NT,)
xB, xD, yB, MB, MD
V, B, R, D
erintB, erintD

State variables:
---
M*x_L; NT+2
M; NT
erintD: ∫eD dt, integral of the distillate composition error
erintB: ∫eB dt, integral of the bottom composition error

State vector: y shape(2NT+4,) float64
---
y[0 : NT+2] : stage holdup of the light component (M*x_L), mol
    y[0] : bottom
    y[1:NT+1] : trays 1 to NT
    y[NT+1] : reflux drum
y[NT+2:2NT+2] : liquid holdup on each tray (M), mol
y[2NT+2] : erintD, min
y[2NT+3] : erintB, min
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

@dataclass(frozen=True)
class ColumnBinaryParams:
    """Parameters from Table 5.7."""
    # Column
    NT: int = 20
    NF: int = 10
    # Feed
    F: float = 100.0       # mol/min
    z: float = 0.55
    # Hydraulic
    M0: float = 10.0       # mol
    beta: float = 0.1      # min
    # Properties
    alpha: float = 2.0
    # PI controller
    R0: float = 128.01     # mol/min
    V0: float = 178.01     # mol/min
    KcD: float = 1000.0    # mol/min
    KcB: float = 1000.0    # mol/min
    xD_set: float = 0.98
    xB_set: float = 0.02
    tauD: float = 5.0      # min
    tauB: float = 1.25     # min
    # Level controller
    MD: float = 100.0      # mol
    MB: float = 100.0      # mol


class ColumnOutputs(NamedTuple):
    """
    Return value of outputs(): all the algebraic variables plus the holdups.
    MB, MD are included so that index n refers to stage n in every array.
    """
    x_L: np.ndarray        # (NT+2,) x_L[0] = xB, x_L[NT+1] = xD
    y_V: np.ndarray        # (NT+1,) y_V[0] = yB, total condenser
    M: np.ndarray          # (NT+2,) M[0] = MB, M[NT+1] = MD, mol
    L: np.ndarray          # (NT+2,) L[0] = B, L[NT+1] = R, mol/min
    V: float               # mol/min
    B: float               # mol/min
    R: float               # mol/min
    D: float               # mol/min


def equilibrium(x_L, alpha):
    """
    x_L is an array, return y_V is also an array with the same shape
    """
    return alpha * x_L / (1 + (alpha - 1) * x_L)


def hydraulics(L0, M, p):
    """
    Calculate the liquid flow rate of all trays (NT,)
    L0, M are arrays, return L is also an array with the same shape
    """
    return L0 + (M - p.M0) / p.beta


def pack(x_L, M, erintD, erintB, p):
    """
    Pack the variables into the state vector y
    x_L: (NT+2,) liquid mole fraction
    M: (NT+2,) liquid holdups, mol. Only the tray holdups M[1:NT+1] are used.
    """
    NT = p.NT

    x_L = np.array(x_L, dtype=np.float64)
    M = np.array(M, dtype=np.float64)
    erint = np.array((erintD,erintB), dtype=np.float64)
    
    M_only_tray = M[1:NT + 1]
    x_L_only_tray = x_L[1:NT + 1]
    Mx_L_only_tray = M_only_tray * x_L_only_tray
    Mx_L = np.append(np.append(p.MD * x_L[0], Mx_L_only_tray), p.MB * x_L[-1])
    
    y = np.concatenate((Mx_L, M_only_tray, erint))

    return y


def unpack(y, p):
    """
    Unpack the state vector y into x_L, M, erintD, erintB
    x_L: (NT+2,) liquid mole fraction
    M: (NT+2,) liquid holdups, mol. M[0] and M[NT+2] comes from p.MD, p.MB
    """
    NT = p.NT
    
    Mx_L = y[0:NT + 2]
    M_only_tray = y[NT + 2:2 * NT + 2]
    erintD = y[2 * NT + 2]
    erintB = y[2 * NT + 3]

    M = np.append(np.append(p.MD, M_only_tray), p.MB)
    x_L = Mx_L / M

    return x_L, M, erintD, erintB


def outputs(y, p):
    """Calculate all the algebraic variables and return ColumnOutputs"""
    NT, NF = p.NT, p.NF
    R0, V0, F = p.R0, p.V0, p.F
    KcD, KcB = p.KcD, p.KcB
    xD_set, xB_set = p.xD_set, p.xB_set
    tauD, tauB = p.tauD, p.tauB

    # unpack
    x_L, M, erintD, erintB = unpack(y, p)

    # Equilibrium
    y_V = equilibrium(x_L, p.alpha)[:-1]

    # PI controller
    eD = xD_set - x_L[-1]
    eB = xB_set - x_L[0]
    R = R0 + KcD * (eD + erintD / tauD)
    V = V0 - KcB * (eB + erintB / tauB)

    # Level controller
    D = V - R
    L1 = hydraulics(R0 + F, M[1], p)
    B = L1 - V

    # Hydraulics
    L0 = np.empty(NT)
    L0[0:NF] = R0 + F
    L0[NF:NT] = R0
    L_only_tray = hydraulics(L0, M[1:NT + 1], p) 
    L = np.append(np.append(B, L_only_tray), R)

    return ColumnOutputs(x_L=x_L, y_V=y_V, M=M, L=L, V=V, B=B, R=R, D=D)


def f(t, y, p):
    """
    Calculate the derivatives of the state variables explicitly
    """
    NT, NF = p.NT, p.NF
    F, z = p.F, p.z
    xD_set, xB_set = p.xD_set, p.xB_set
    o = outputs(y, p)
    L, x_L, y_V = o.L, o.x_L, o.y_V
    V, R, D, B = o.V, o.R, o.D, o.B

    # liquid holdup on each tray
    dM = L[2:NT + 2] - L[1:NT + 1]
    dM[NF - 1] += F #feed
    
    # holdup of the light component on each tray
    dMx_L_tray = L[2:NT + 2] * x_L[2:NT + 2] - L[1:NT + 1] * x_L[1:NT + 1] + V * (y_V[0:NT] - y_V[1:NT + 1])

    # drum
    dMDxD = V * y_V[NT] - R * x_L[NT + 1] - D * x_L[NT + 1]

    # base
    dMBxB = L[1] * x_L[1] - V * y_V[0] - B * x_L[0]

    dMx_L = np.append(np.append(dMBxB, dMx_L_tray), dMDxD)
    dMx_L[NF] += F * z #feed
    
    eD = xD_set - x_L[NT + 1]
    eB = xB_set - x_L[0]

    dy = np.concatenate((dMx_L, dM, np.array((eD, eB))))

    return dy

def steady_state(p):
    pass