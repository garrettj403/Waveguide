"""Tests for cavity sub-module."""

import numpy as np
import scipy.constants as sc
from pytest import approx

import waveguide as wg


def test_example_6p1():
    """Test example 6.1 in Pozar."""

    # Conductivity
    cond = 5.813e7

    # Frequency
    freq = 5 * sc.giga

    # Surface resistance
    rs = wg.surface_resistance(freq, cond, ur=1)
    assert rs == approx(1.84e-2, abs=0.02e-2)


def test_example_6p3():
    """Test example 6.3 in Pozar."""

    # H-band waveguide dimensions
    a = 4.755 * sc.centi
    b = 2.215 * sc.centi
    d = 2.20 * sc.centi

    # Conductivity
    cond = 5.813e7

    # Mode numbers
    m, n = 1, 0
    l = 1

    # Dielectric properties of polyethylene
    er_mag = 2.25
    tand = 4e-4
    ur = 1

    # Wavenumber
    f = 5 * sc.giga
    k = wg.wavenumber(f, er=er_mag, ur=ur)
    k2 = 2 * sc.pi * f * np.sqrt(er_mag) / sc.c
    assert k == approx(k2)
    assert k == approx(157.08, abs=1)

    # Resonant frequency
    fres = wg.resonant_frequency(a, b, d, m=m, n=n, l=l, er=er_mag, ur=1)
    assert fres == approx(f, abs=0.02*sc.giga)

    # Intrinsic impedance
    eta = wg.intrinsic_impedance(er=er_mag, ur=ur)
    assert eta == approx(251.3, abs=0.2)

    # Conduction Q-factor
    rs = wg.surface_resistance(f, cond, ur=ur)
    assert rs == approx(1.84e-2, abs=0.02e-2)
    qc = wg.qfactor_conduction(a, b, d, cond, m=m, n=n, l=1, er=er_mag, ur=ur)
    assert qc == approx(8403, abs=10)

    # Dielectric Q-factor
    qd = wg.qfactor_dielectric(tand)
    assert qd == approx(2500, abs=2)

    # Parallel Q-factor
    q_net = wg.qfactor_parallel(qc, qd)
    assert q_net == approx(1927, abs=3)


def test_problem_6p9():

    # X-band waveguide dimensions
    a = 2.286 * sc.centi
    b = 1.016 * sc.centi
    d = 2.0 * sc.centi

    # Conductivity of aluminum
    cond = 3.816e7

    # Resonance 101
    f101 = wg.resonant_frequency(a, b, d, m=1, n=0, l=1, er=1, ur=1)
    rs = wg.surface_resistance(f101, cond, ur=1)
    k = wg.wavenumber(f101, er=1, ur=1)
    q101 = wg.qfactor_conduction(a, b, d, cond, m=1, n=0, l=1, er=1, ur=1)
    assert f101 == approx(9.965 * sc.giga, abs=0.015*sc.giga)
    assert rs == approx(0.0321, abs=0.0002)
    assert k == approx(208.7, abs=0.2)
    assert q101 == approx(6349, abs=10)

    # Resonance 102
    f102 = wg.resonant_frequency(a, b, d, m=1, n=0, l=2, er=1, ur=1)
    rs = wg.surface_resistance(f102, cond, ur=1)
    k = wg.wavenumber(f102, er=1, ur=1)
    q102 = wg.qfactor_conduction(a, b, d, cond, m=1, n=0, l=2, er=1, ur=1)
    assert f102 == approx(16.372 * sc.giga, abs=0.015 * sc.giga)
    assert rs == approx(0.0412, abs=0.0002)
    assert k == approx(342.9, abs=0.2)
    assert q102 == approx(7987, abs=10)


# TODO: add problem 6.23 from Pozar


if __name__ == "__main__":

    test_example_6p1()
    test_example_6p3()
    test_problem_6p9()
