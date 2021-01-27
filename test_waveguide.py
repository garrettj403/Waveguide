import pytest
import waveguide as wg
import scipy.constants as sc


def test_eta():
    """Test intrinsic impedance of vacuum."""

    value = wg.intrinsic_impedance(1, 1)
    scipy_value = sc.physical_constants['characteristic impedance of vacuum'][0]
    assert value == pytest.approx(scipy_value)


def test_example_3p1():
    """Test example 3.1 in Pozar."""

    # Dimensions
    a, b = 1.07*sc.centi, 0.43*sc.centi

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
    assert te10 == pytest.approx( 9.72 * sc.giga, 0.01 * sc.giga)
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
