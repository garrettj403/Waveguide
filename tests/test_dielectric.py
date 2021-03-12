"""Tests for dielectric sub-module."""

import numpy as np 
import scipy.constants as sc
import skrf as rf

import waveguide as wg


def test_against_scikit_rf():
    """Compare against Scikit-RF."""

    # WR-28
    a, b = 0.28*sc.inch, 0.14*sc.inch

    # Conductivity
    cond = 1.8e7

    # Frequency sweep
    freq = rf.Frequency()
    freq.f = np.linspace(22, 42, 201) * sc.giga

    # Relativity permittivity
    er = 9.3

    # Length
    length = 1 * sc.inch

    # Scikit-RF
    wr28_air     = rf.RectangularWaveguide(freq, a=a, b=b, rho=1/cond, ep_r=1.)
    wr28_alumina = rf.RectangularWaveguide(freq, a=a, b=b, rho=1/cond, ep_r=er)

    # Waveguide length
    total_length = 1.7*sc.inch
    length1 = (total_length - length) / 2
    length2 = length
    length3 = length1

    # Build network
    wg1 = wr28_air.line(length1, unit='m')
    wg2 = wr28_alumina.line(length2, unit='m')
    wg3 = wr28_air.line(length3, unit='m')
    network = wg1 ** wg2 ** wg3

    # Waveguide package
    _, _, s21, _ = wg.dielectric_sparam(freq.f, a, b, er, 0, cond, length1, length2, length3)

    # Compare
    np.testing.assert_almost_equal(network.s_db[:,1,0], 20*np.log10(np.abs(s21)), decimal=2)
