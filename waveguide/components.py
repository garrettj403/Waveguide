"""Various waveguide components."""

import scipy.constants as sc


def thin_iris(wg, diameter):
    """Create circular iris Scikit-RF component.

    Args:
        wg (skrf.RectangularWaveguide): rectangular waveguide
        diameter (float): diameter of circular iris, in [m]

    Returns:
        skrf.RectangularWaveguide.line: iris

    """

    radius = diameter / 2

    # Calculate iris inductance
    alpha_m = 4 * radius ** 3 / 3
    b_norm_iris = -wg.a * wg.b / (2 * wg.beta * alpha_m)
    b_iris = b_norm_iris / wg.z0.real
    l_iris = -1 / b_iris / (2 * sc.pi * wg.frequency.f)

    # Create iris component
    iris = wg.shunt_inductor(l_iris)
    iris.name = "Iris: d={:.2f}mm".format(diameter*1e3)

    return iris
