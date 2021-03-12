"""Dielectric-filled waveguide."""

import numpy as np

from waveguide import propagation_constant, impedance


def dielectric_sparam(f, a, b, er_mag, tand, cond, length1, length2, length3):
    """Calculate the S-parameters of a waveguide containing a section of
    dielectric material.

    | Section | Material   |
    | ------- | ---------- |
    | 1       | Air        |
    | 2       | Dielectric |
    | 3       | Air        |

    Args:
        f: frequency
        a: broad waveguide dimension
        b: narrow waveguide dimension
        er_mag: relative permittivity magnitude of dielectric
        tand: loss tangent of dielectric
        cond: conductivity of waveguide walls
        length1: length of empty waveguide between port 1 and the dielectric
        length2: length of dielectric-filled waveguide
        length3: length of empty waveguide between port 2 and the dielectric

    Returns:
        tuple: S-parameters: S11, S22, S21, and S21 of an empty waveguide

    """

    # Dielectric properties
    er = er_mag * (1 - 1j * tand)
    ur = 1

    # Propagation constant
    gamma1 = propagation_constant(f, a, b=b, er=1., ur=1., cond=cond, m=1, n=0)
    gamma2 = propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=1, n=0)
    gamma3 = gamma1.copy()

    # Transmission coefficient
    r1 = np.exp(-gamma1 * length1)  # empty waveguide connected to port 1
    r2 = np.exp(-gamma2 * length2)  # dielectric filled waveguide
    r3 = np.exp(-gamma3 * length3)  # empty waveguide connected to port 2

    # Reflection coefficient between air-filled and dielectric-filled waveguide
    zte1 = impedance(f, a, b=b, er=1., ur=1., cond=cond, m=1, n=0, mode='TE')
    zte2 = impedance(f, a, b=b, er=er, ur=ur, cond=cond, m=1, n=0, mode='TE')
    reflec = (zte1 - zte2) / (zte1 + zte2)

    # S-parameters
    s11 = r1 ** 2 * (reflec * (1 - r2 ** 2) / (1 - reflec ** 2 * r2 ** 2))
    s22 = r3 ** 2 * (reflec * (1 - r2 ** 2) / (1 - reflec ** 2 * r2 ** 2))
    s21 = r1 * r3 * (r2 * (1 - reflec ** 2) / (1 - reflec ** 2 * r2 ** 2))
    s21_air = r1 * r3 * np.exp(-gamma1 * length2)

    return s11, s22, s21, s21_air
