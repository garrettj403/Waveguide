"""Waveguide cavity resonator."""

import numpy as np
from numpy import pi, sqrt
from scipy.constants import c as c0


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

    return c0 / 2 / pi / sqrt(er.real * ur.real) * sqrt((m*pi/a)**2 + (n*pi/b)**2 + (l*pi/d)**2)
