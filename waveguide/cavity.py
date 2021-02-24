"""Waveguide cavity resonator."""

import numpy as np
from numpy import pi, sqrt
from scipy.constants import c as c0
import scipy.constants as sc

from .propagation import surface_resistance, wavenumber, intrinsic_impedance


# Free space impedance
eta0 = sc.physical_constants['characteristic impedance of vacuum'][0]


def resonant_frequency(a, b, d, m=1, n=0, l=0, er=1, ur=1):
    """Calculate the resonant frequencies of a waveguide cavity.

    Args:
        a: a waveguide dimension
        b: b waveguide dimension
        d: length of waveguide cavity
        m: mode number m
        n: mode number n
        l: resonance number l
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: resonant frequency

    """

    term1 = c0 / 2 / pi / sqrt(er.real * ur.real)
    term2 = sqrt((m * pi / a) ** 2 + (n * pi / b) ** 2 + (l * pi / d) ** 2)

    return term1 * term2


# Q-FACTOR ---------------------------------------------------------------- ###

def qfactor_dielectric(tand):
    """Calculate Q-factor due to dielectric filling.

    Args:
        tand: loss tangent

    Returns:
        Q-factor

    """

    return 1 / tand


def qfactor_conduction(a, b, d, cond, m=1, n=0, l=1, er=1, ur=1):
    """Calculate Q-factor due to waveguide conductivity.

    Args:
        a: a waveguide dimension
        b: b waveguide dimension
        d: length of waveguide cavity
        cond: conductivity
        m: mode number m
        n: mode number n
        l: resonance number l
        er: relative permittivity
        ur: relative permeability

    Returns:
        Q-factor

    """

    # Resonant frequency
    fres = resonant_frequency(a, b, d, m, n, l, er=er, ur=ur)

    # Surface resistance
    rs = surface_resistance(fres, cond)

    # Wavenumber (1/m)
    k = wavenumber(fres, er=er, ur=ur)

    # Intrinsic impedance
    eta = intrinsic_impedance(er=er, ur=ur)

    # Eqn 6.46 in Pozar
    t1 = (k * a * d) ** 3 * b * eta / 2 / pi ** 2 / rs
    t2 = 2 * l ** 2 * a ** 3 * b + 2 * b * d ** 3 + l ** 2 * a ** 3 * d + a * d ** 3

    return t1 / t2


def qfactor_parallel(q1, q2):
    """Parallel Q-factors.

    Args:
        q1 (float): Q-factor 1
        q2 (float): Q-factor 2

    Returns:
        float: parallel Q-factor

    """

    return q1 * q2 / (q1 + q2)
