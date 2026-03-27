"""Base rate constant computation.

Surface reactions — harmonic transition state theory (hTST):
    k_f = (kB*T / h) * exp(-Ea / (kB*T))
    k_r = k_f / K_eq   where K_eq = exp(-DeltaE / (kB*T))

Adsorption — collision theory (S=1, non-activated):
    k_f = 1 / (N_sites * sqrt(2*pi*m*kB_SI*T))
    k_r = k_f / K_eq   where K_eq = exp(-DeltaE / (kB*T))

This module computes base rate constants only.
Lateral interactions and ODE assembly live in build_mkm.py.
"""

import numpy as np
from typing import Tuple

from mkm_solver.utils.constants import KB, H, KB_SI, AMU_TO_KG


def rate_constants_surface(
    Ea: float, DeltaE: float, T: float
) -> Tuple[float, float]:
    """hTST rate constants for a surface reaction.
    Returns (k_forward, k_reverse).
    """
    kBT = KB * T
    prefactor = kBT / H
    kf = prefactor * np.exp(-Ea / kBT)
    K_eq = np.exp(-DeltaE / kBT)
    kr = kf / K_eq
    return kf, kr


def rate_constants_adsorption(
    DeltaE: float, T: float,
    mass_amu: float, n_sites: float,
) -> Tuple[float, float]:
    """Collision theory rate constants for adsorption.
    k_f = 1 / (N_sites * sqrt(2*pi*m*kB_SI*T))   [S=1, non-activated]
    k_r = k_f / K_eq
    Returns (k_forward, k_reverse).
    """
    mass_kg = mass_amu * AMU_TO_KG
    kf = 1.0 / (n_sites * np.sqrt(2.0 * np.pi * mass_kg * KB_SI * T))
    K_eq = np.exp(-DeltaE / (KB * T))
    kr = kf / K_eq
    return kf, kr
