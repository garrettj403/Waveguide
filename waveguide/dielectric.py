"""Dielectric-filled waveguide."""

import numpy as np

from waveguide import propagation_constant, impedance


def dielectric_sparam(f, a, b, er_mag, tand, cond, length1, length2, total_length):
    """Calculate the S-parameters of a waveguide containing a section of
    dielectric material.

    Args:
        f: frequency
        a: broad waveguide dimension
        b: narrow waveguide dimension
        er_mag: relative permittivity magnitude of dielectric
        tand: loss tangent of dielectric
        cond: conductivity of waveguide walls
        length1: length of empty waveguide between port 1 and the dielectric
        length2: length of empty waveguide between port 2 and the dielectric
        total_length: total length of the waveguide

    Returns:
        tuple: S-parameters: S11, S22, S21, and S21 of an empty waveguide

    """

    # Dielectric properties
    er = er_mag * (1 - 1j * tand)
    ur = 1

    # Propagation constant (0 denotes empty waveguide)
    gamma0 = propagation_constant(f, a, b=b, er=1, ur=1, cond=cond, m=1, n=0)
    gamma = propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=1, n=0)

    # Length of dielectric
    length_d = total_length - length1 - length2

    # Transmission coefficient
    r1 = np.exp(-gamma0 * length1)  # empty waveguide connected to port 1
    r2 = np.exp(-gamma0 * length2)  # empty waveguide connected to port 2
    z = np.exp(-gamma * length_d)  # dielectric filled waveguide

    # Reflection coefficient between air-filled and dielectric-filled waveguide
    zte0 = impedance(f, a, b=b, er=1, ur=1, cond=cond, m=1, n=0, mode='TE')
    zte = impedance(f, a, b=b, er=er, ur=ur, cond=cond, m=1, n=0, mode='TE')
    reflec = (zte0 - zte) / (zte0 + zte)

    # S-parameters
    s11 = r1 ** 2 * (reflec * (1 - z ** 2) / (1 - reflec ** 2 * z ** 2))
    s22 = r2 ** 2 * (reflec * (1 - z ** 2) / (1 - reflec ** 2 * z ** 2))
    s21 = r1 * r2 * (z * (1 - reflec ** 2) / (1 - reflec ** 2 * z ** 2))
    s21_0 = r1 * r2 * np.exp(-gamma0 * length_d)

    return s11, s22, s21, s21_0
