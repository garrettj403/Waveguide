"""Tests for cavity sub-module."""

import os
import numpy as np
import scipy.constants as sc
from pytest import approx
import matplotlib.pyplot as plt
import skrf as rf

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
    ell = 1

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
    fres = wg.resonant_frequency(a, b, d, m=m, n=n, l=ell, er=er_mag, ur=1)
    assert fres == approx(f, abs=0.02*sc.giga)

    # Recover permittivity
    er_mag_recovered = wg.resonant_frequency2permittivity(ell, fres, a, b, d)
    assert er_mag_recovered == approx(er_mag, abs=0.02)

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

    # De-embed Q-factor
    qc_recovered = wg.deembed_qfactor(q_net, qd)
    assert qc_recovered == approx(qc, abs=10)

    # Recover surface resistance
    rs_recovered = wg.q2surface_resistance(fres, qc, a, b, d, l=ell, er=er_mag, ur=ur)
    assert rs_recovered == approx(rs, abs=2e-4)

    # Recover conductivity
    cond_recovered = wg.q2conductivity(fres, qc, a, b, d, l=ell, er=er_mag, ur=ur)
    assert cond_recovered == approx(cond, 2e4)

    # Recover loss tangent
    tand_recovered = wg.q2loss_tangent(qd)
    assert tand_recovered == approx(tand, abs=2e-4)


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


def test_problem_6p23():
    """Test problem 6.23 from Pozar."""

    fres = 9 * sc.giga
    q = 11_000
    a, b = 2.5*sc.centi, 1.25*sc.centi

    # Wavenumber
    k0 = wg.wavenumber(fres, er=1, ur=1)
    assert k0 == approx(188, abs=5)

    # Phase constant
    beta0 = wg.phase_constant(fres, a, b, er=1, ur=1, m=1, n=0)
    assert beta0 == approx(140.5, abs=0.5)

    # Length
    length = np.pi / beta0
    assert length == approx(2.24 * sc.centi, abs=0.05 * sc.centi)


def test_simulated_cavity(debug=False):

    # Dimensions
    a, b, d = 280*sc.mil, 140*sc.mil, 6*sc.inch
    cond = 1e7

    # Load simulated data
    filename = os.path.join('data', 'cavity.s2p')
    dir_name = os.path.dirname(__file__)
    filename = os.path.join(dir_name, filename)
    data = rf.Network(filename)

    # Unpack
    f = data.f
    s21 = data.s[:, 1, 0]

    # Find resonances
    fres_list = wg.find_resonances(f, wg.db20(s21), height=-90)

    # Get Q-factor
    fres, q0, ql, _ = wg.find_qfactor(f, np.abs(s21), fres_list, fspan=5e7, ncol=6, figsize=(14,8), debug=debug)

    # Resonant frequencies (theory)
    ell_start = 3
    ell = np.arange(ell_start, ell_start + len(fres))
    fres_theory = wg.resonant_frequency(a, b, d, l=ell)
    np.testing.assert_almost_equal(fres / 1e9, fres_theory / 1e9, decimal=1)

    # Q-factor (theory)
    qc_theory = wg.qfactor_conduction(a, b, d, cond, l=ell)

    # Plot Q-factor
    if debug:
        plt.figure()
        plt.plot(ell, qc_theory, 'ko--', label='Theory')
        plt.plot(ell, q0, 'ro--', label="Q-factor (corrected)")
        plt.plot(ell, ql, 'bo--', label="Q-factor (loaded)")
        plt.xlim(xmin=0)
        plt.legend()
        plt.show()

    # Get conductivity from Q-factor
    cond_q = wg.q2conductivity(fres, q0, a, b, d, l=ell)
    cond_theory = wg.q2conductivity(fres_theory, qc_theory, a, b, d, l=ell)

    # Plot conductivity
    if debug:
        plt.figure()
        plt.plot(fres, cond_q, 'bo-', label="From Q-factor")
        plt.plot(fres_theory, cond_theory, 'ro--', label="From theory")
        plt.axhline(cond, c='k', ls='--')
        plt.ylabel("Conductivity (S/m)")
        plt.xlabel("Frequency (GHz)")
        plt.show()

    # Test
    np.testing.assert_allclose(cond_q, cond*np.ones_like(cond_q), atol=1e6)
    np.testing.assert_allclose(cond_theory, cond * np.ones_like(cond_q), atol=1)


# def test_simulated_cavity_with_hdpe(debug=False):
#
#     # Dimensions
#     a, b, d = 280*sc.mil, 140*sc.mil, 6*sc.inch
#     cond = 1e7
#     er
#
#     # Load simulated data
#     filename = os.path.join('data', 'cavity.s2p')
#     dir_name = os.path.dirname(__file__)
#     filename = os.path.join(dir_name, filename)
#     data = rf.Network(filename)
#
#     # Unpack
#     f = data.f
#     s21 = data.s[:, 1, 0]
#
#     # Find resonances
#     fres_list = wg.find_resonances(f, wg.db20(s21), height=-90)
#
#     # Get Q-factor
#     fres, q0, ql, _ = wg.find_qfactor(f, np.abs(s21), fres_list, fspan=5e7, ncol=6, figsize=(14,8), debug=debug)
#
#     # Resonant frequencies (theory)
#     ell = np.arange(3, 3 + len(fres))
#     fres_theory = wg.resonant_frequency(a, b, d, l=ell)
#     np.testing.assert_almost_equal(fres / 1e9, fres_theory / 1e9, decimal=1)
#
#     # Q-factor (theory)
#     qc_theory = wg.qfactor_conduction(a, b, d, cond, l=ell)
#
#     # Plot Q-factor
#     if debug:
#         plt.figure()
#         plt.plot(ell, q0, 'bo--', label="Q-factor (corrected)")
#         plt.plot(ell, ql, 'ro--', label="Q-factor (loaded)")
#         plt.plot(ell, qc_theory, 'ko--', label='Theory')
#         plt.xlim(xmin=0)
#         plt.legend()
#         plt.show()
#
#     # Get conductivity from Q-factor
#     cond_q = wg.q2conductivity(fres, q0, a, b, d, l=ell)
#     cond_theory = wg.q2conductivity(fres_theory, qc_theory, a, b, d, l=ell)
#
#     # Plot conductivity
#     if debug:
#         plt.figure()
#         plt.plot(fres, cond_q, 'bo-', label="From Q-factor")
#         plt.plot(fres_theory, cond_theory, 'ro--', label="From theory")
#         plt.axhline(cond, c='k', ls='--')
#         plt.ylabel("Conductivity (S/m)")
#         plt.xlabel("Frequency (GHz)")
#         plt.show()
#
#     # Test
#     np.testing.assert_allclose(cond_q, cond*np.ones_like(cond_q), atol=1e6)
#     np.testing.assert_allclose(cond_theory, cond * np.ones_like(cond_q), atol=1)


if __name__ == "__main__":

    # test_example_6p1()
    # test_example_6p3()
    # test_problem_6p9()
    # test_problem_6p23()
    test_simulated_cavity(debug=True)
