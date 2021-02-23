"""Various utilities."""

import numpy as np
from numpy import log, log10, pi


def db2np(value):
    """Convert dB value to Np."""
    return value * log(10) / 20


def np2db(value):
    """Convert Np value to dB."""
    return value * 20 / log(10)


def db10(value):
    """Convert power-like value to dB."""
    return 10 * log10(np.abs(value))


def db20(value):
    """Convert voltage-like value to dB."""
    return 20 * log10(np.abs(value))


def rad2deg(value):
    """Convert radians to degrees."""
    return value * 180 / pi


def deg2rad(value):
    """Convert degrees to radians."""
    return value / 180 * pi
