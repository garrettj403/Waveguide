"""Tests for propagation sub-module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.constants as sc
from pytest import approx

import waveguide as wg


def test_example_1p2():
    """Test example 1.2 in Pozar."""

    # Frequency
    f = 10 * sc.giga

    # Conductivities
    cond_au = 4.098e7
    cond_ag = 6.173e7
    cond_al = 3.816e7
    cond_cu = 5.813e7

    # Skin depth
    assert wg.skin_depth(f, cond_au) == approx(7.86e-7, abs=0.1e-7)
    assert wg.skin_depth(f, cond_ag) == approx(6.40e-7, abs=0.1e-7)
    assert wg.skin_depth(f, cond_al) == approx(8.14e-7, abs=0.1e-7)
    assert wg.skin_depth(f, cond_cu) == approx(6.60e-7, abs=0.1e-7)


def test_example_3p1():
    """Test example 3.1 in Pozar."""

    # Dimensions
    a, b = 1.07 * sc.centi, 0.43 * sc.centi

    # Teflon
    er_mag, tand, ur = 2.08, 0.0004, 1
    er = er_mag * (1 - 1j * tand)

    # Copper waveguide walls
    cond = 5.8e7

    # Test cutoff frequencies
    te10 = wg.cutoff_frequency(a, b, er_mag, 1, 1, 0)
    te20 = wg.cutoff_frequency(a, b, er_mag, 1, 2, 0)
    te01 = wg.cutoff_frequency(a, b, er_mag, 1, 0, 1)
    te11 = wg.cutoff_frequency(a, b, er_mag, 1, 1, 1)
    te21 = wg.cutoff_frequency(a, b, er_mag, 1, 2, 1)
    assert te10 == pytest.approx(9.72 * sc.giga, 0.01 * sc.giga)
    assert te20 == pytest.approx(19.44 * sc.giga, 0.01 * sc.giga)
    assert te01 == pytest.approx(24.19 * sc.giga, 0.01 * sc.giga)
    assert te11 == pytest.approx(26.07 * sc.giga, 0.01 * sc.giga)
    assert te21 == pytest.approx(31.03 * sc.giga, 0.01 * sc.giga)

    # Properties at 15 GHz...
    f = 15 * sc.giga
    k = wg.wavenumber(f, er_mag, ur)
    alphad = wg.dielectric_loss(f, a, b, er=er, ur=ur, m=1, n=0)
    beta = wg.phase_constant(f, a, b, er=er, ur=ur, cond=cond, m=1, n=0)
    rs = wg.surface_resistance(f, cond, ur=ur)
    alphac = wg.conductor_loss(f, cond, a, b, er=er_mag, ur=ur)
    assert k == pytest.approx(453.1, 0.1)
    assert beta == pytest.approx(345.1, 0.1)
    assert alphad == pytest.approx(0.119, 0.001)
    assert alphac == pytest.approx(0.050, 0.002)
    assert rs == pytest.approx(0.032, 0.002)


def test_example_4p2():
    """Test example 4.2 in Pozar."""

    # X-band waveguide dimensions
    a = 2.285 * sc.centi
    b = 1.016 * sc.centi

    # Rexolite
    er_mag = 2.54

    # Frequency
    f = 10 * sc.giga

    # Propagation constants
    beta_a = wg.phase_constant(f, a, b, er=1)
    beta_d = wg.phase_constant(f, a, b, er=er_mag)
    assert beta_a == approx(158.0, abs=0.5)
    assert beta_d == approx(304.1, abs=0.5)

    # Characteristic impedance
    z0_a = wg.impedance(f, a, b, er=1)
    z0_d = wg.impedance(f, a, b, er=er_mag)
    print(z0_a)
    print(z0_d)
    assert z0_a == approx(500.0, abs=1)
    assert z0_d == approx(259.6, abs=1)


def test_problem_3p5():
    """Test problem 3.5 in Pozar."""

    # Dimensions
    a, b = 1.07*sc.centi, 0.43*sc.centi
    length = 10*sc.centi

    # Teflon
    er_mag, tand, ur = 2.55, 0.0015, 1
    er = er_mag * (1 - 1j * tand)

    # Copper waveguide walls
    cond = 5.8e7

    # Test cutoff frequencies
    te10 = wg.cutoff_frequency(a, b, er_mag, 1, 1, 0)
    te20 = wg.cutoff_frequency(a, b, er_mag, 1, 2, 0)
    assert te10 == pytest.approx(8.78 * sc.giga, 0.01 * sc.giga)
    assert te20 == pytest.approx(17.6 * sc.giga, 0.1 * sc.giga)

    # Properties at 15 GHz...
    f = 15 * sc.giga
    k = wg.wavenumber(f, er_mag, ur)
    alphad = wg.dielectric_loss(f, a, b, er=er, ur=ur, m=1, n=0)
    beta = wg.phase_constant(f, a, b, er=er, ur=ur, cond=cond, m=1, n=0)
    rs = wg.surface_resistance(f, cond, ur=ur)
    alphac = wg.conductor_loss(f, cond, a, b, er=er_mag, ur=ur)
    alpha = wg.propagation_constant(f, a, b, er=er, ur=ur, cond=cond, m=1, n=0).real
    total_loss = wg.np2db(alpha) * length
    phase = wg.rad2deg(beta * length)
    assert k == pytest.approx(501.67, 0.01)
    assert beta == pytest.approx(406.78, 0.01)
    assert alphad == pytest.approx(0.464, 0.001)
    assert alphac == pytest.approx(0.0495, 0.0002)
    assert total_loss == pytest.approx(0.446, 0.001)
    assert rs == pytest.approx(0.03195, 0.0001)
    assert phase == pytest.approx(2330.7, 0.2)


def test_eta():
    """Test intrinsic impedance of vacuum against known value."""

    val = wg.intrinsic_impedance(1, 1)
    scipy_val = sc.physical_constants['characteristic impedance of vacuum'][0]
    assert val == pytest.approx(scipy_val)


def test_dielectric_loss(debug=False):

    # Dielectric properties
    er_mag = 3
    tand = 1e-2
    er = er_mag * (1 - 1j * tand)

    # Waveguide dimensions
    a, b = 28 * sc.mil, 14 * sc.mil

    # Waveguide mode
    m, n = 1, 0

    # Frequency sweep
    f = np.linspace(260, 400, 141) * 1e9
    w = 2 * np.pi * f

    # From propagation constant
    alphad1 = wg.dielectric_loss(f, a, b=b, er=er, ur=1, m=m, n=n)

    # From Eqn. 3.29 in Pozar
    k = wg.wavenumber(f, er=er.real, ur=1)
    beta = wg.phase_constant(f, a, b=b, er=er, ur=1, m=m, n=n)
    alphad2 = k ** 2 * tand / 2 / beta

    # Debug
    if debug:
        plt.figure()
        plt.plot(f/1e9, alphad1, 'k')
        plt.plot(f/1e9, alphad2, 'r--')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Attenuation Constant (Np/m)")
        plt.show()

    # Compare
    np.testing.assert_almost_equal(alphad1, alphad2, decimal=12)


def test_conductor_loss(debug=False):
    """Compare to model from Maxwell's 1947 paper."""

    # Waveguide conductivity [S/m]
    cond = 1e7

    # Waveguide dimensions
    a, b = 28 * sc.mil, 14 * sc.mil

    # Waveguide mode
    m, n = 1, 0

    # Frequency sweep
    f = np.linspace(260, 400, 141) * 1e9
    w = 2 * np.pi * f

    # From Eqn. 3.96 in Pozar
    alphac1 = wg.conductor_loss(f, cond, a, b, er=1, ur=1)

    # From Maxwell 1947
    k = wg.wavenumber(f, er=1, ur=1)
    beta = wg.phase_constant(f, a, b=b, er=1, ur=1, m=m, n=n)
    lambdac = 2 * a
    lambda0 = sc.c / f
    alphac2 = 1 / (2 * b) / \
              np.sqrt(1 - (lambda0/lambdac)**2) * \
              np.sqrt(4 * np.pi / lambda0 / sc.mu_0 / sc.c / cond) * \
              (1 + 2 * b / a * (lambda0 / lambdac)**2)

    # Debug
    if debug:
        plt.figure()
        plt.plot(f/1e9, alphac1, 'k')
        plt.plot(f/1e9, alphac2, 'r--')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Attenuation Constant (Np/m)")
        plt.show()

    # Compare
    np.testing.assert_almost_equal(alphac1, alphac2, decimal=12)


def test_effective_conductivity(debug=False):

    # Frequency sweep
    f = np.linspace(260, 400, 141) * 1e9

    # Waveguide conductivity [S/m]
    cond = 1e7 * np.ones_like(f)
    cond += f * 1e-5

    # Waveguide dimensions
    a, b = 28 * sc.mil, 14 * sc.mil

    # Waveguide mode
    m, n = 1, 0

    # From Eqn. 3.96 in Pozar
    alphac = wg.conductor_loss(f, cond, a, b, er=1, ur=1)
    alphac2 = wg.attenuation_constant(f, a, b, er=1, ur=1, cond=cond, m=m, n=n)
    np.testing.assert_almost_equal(alphac, alphac2, decimal=10)

    # Recover conductivity
    cond_eff = wg.effective_conductivity(f, alphac, a, b, er=1, ur=1)

    # Debug
    if debug:
        plt.figure()
        plt.plot(f/1e9, cond * np.ones_like(f), 'k')
        plt.plot(f/1e9, cond_eff, 'r--')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Conductivity (S/m)")
        plt.show()

    # Compare
    np.testing.assert_almost_equal(cond, cond_eff, decimal=8)


def test_cutoff():

    # WR-28
    a, b = 280*sc.mil, 140*sc.mil

    # Cutoff wavelength for TE10
    lambda_c = 2 * a
    f_c = sc.c / 2 / a

    # Cutoff frequency
    assert wg.cutoff_frequency(a) == approx(f_c)

    # Cutoff wavenumber
    assert 2 * np.pi / wg.cutoff_wavenumber(a) == approx(lambda_c)


if __name__ == "__main__":

    # test_example_3p1()
    test_example_4p2()
    # test_problem_3p5()
    # test_eta()
    # test_dielectric_loss(debug=True)
    # test_conductor_loss(debug=True)
    # test_effective_conductivity(debug=True)
