"""Rectangular waveguide properties.

Note: All units are in SI base units. E.g., all lengths are in [m].

"""

import numpy as np

from numpy import sqrt, pi, log, log10, arctan
from scipy.constants import epsilon_0 as e0
from scipy.constants import c as c0
from scipy.constants import mu_0 as u0


# WAVEGUIDE PROPERTIES --------------------------------------------------- ###

def propagation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate propagation constant (complex value).

    Args:
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: propagation constant

    """

    k = wavenumber(f, er, ur)
    kc = cutoff_wavenumber(a, b, m, n)

    if cond is not None:
        alpha_c = conductor_loss(f, cond, a, b, er=er, ur=ur)
    else:
        alpha_c = 0

    return 1j * sqrt(k ** 2 - kc ** 2) + alpha_c


def phase_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate phase constant (beta).

    Args:
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: phase constant

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n).imag


def wavelength(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate wavelength.

    Args:
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: wavelength

    """

    beta = phase_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n)

    return 2 * pi / beta


def attenuation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate phase constant (alpha).

    Args:
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: attenuation constant

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n).real


def intrinsic_impedance(er=1, ur=1):
    """Calculate intrinsic impedance of dielectric.

    Args:
        er: relative permittivity
        ur: relative permeability

    Returns:
        float: intrinsic impedance

    """

    return sqrt((ur * u0) / (er * e0))


def wavenumber(f, er=1, ur=1):
    """Calculate freespace wavenumber.

    Args:
        f: frequency
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: freespace wavenumber
    """

    w = 2 * pi * f

    return w * sqrt(ur * u0 * er * e0)


def cutoff_wavenumber(a, b=None, m=1, n=0):
    """Calculate cutoff wavenumber of mode TE/TMmn.

    Args:
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        m: mode number m
        n: mode number n

    Returns:
        float: cutoff wavenumber
    """

    if b is None:
        b = a / 2

    return sqrt((m * pi / a)**2 + (n * pi / b)**2)


def cutoff_frequency(a, b=None, er=1, ur=1, m=1, n=0):
    """Calculate cutoff frequency of move TE/TMmn.

    Args:
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        m: mode number m
        n: mode number n

    Returns:
        float: cutoff frequency

    """

    kc = cutoff_wavenumber(a, b, m, n)

    return c0 / (2 * pi * sqrt(er * ur)) * kc


def impedance(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0, mode='TE'):

    k = wavenumber(f, er=er, ur=ur)
    eta = intrinsic_impedance(er=er, ur=ur)
    beta = phase_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n)

    if mode.lower() == 'te':
        return k * eta / beta
    elif mode.lower() == 'tm':
        return beta * eta / k
    else:
        print("Mode must be either TE or TM")
        raise


# WAVEGUIDE LOSS --------------------------------------------------------- ###

def dielectric_loss(f, a, b=None, er=1, ur=1, m=1, n=0):
    """Calculate dielectric loss.

    Args:
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        m: mode number m
        n: mode number n

    Returns:

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=None, m=m, n=n).real


def conductor_loss(f, cond, a, b, er=1, ur=1):
    """Calculate loss due to conduction in waveguide walls.

    Only for TE10 mode.

    Args:
        f: frequency
        cond: conductivity of waveguide walls (~5.8e7 for copper)
        a: broad waveguide dimension
        b: narrow waveguide dimension
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: conductor loss in Np/m

    """

    # Propagation properties
    k = np.real(wavenumber(f, er=er, ur=ur))
    beta = np.imag(propagation_constant(f, a, b, er=er, ur=ur, m=1, n=0))
    eta = np.real(intrinsic_impedance(er=er, ur=ur))

    # Surface resistance
    rs = surface_resistance(f, cond, ur=ur.real)

    # Conductor loss (Eqn. 3.96 in Pozar)
    return rs / (a**3 * b * beta * k * eta) * (2 * b * pi**2 + a**3 * k**2)


def surface_resistance(f, cond, ur=1):
    """Calculate surface resistance.

    Args:
        f: frequency
        cond: conductivity
        ur: relative permeability

    Returns:
        np.ndarray: surface resistance

    """

    w = 2 * pi * f

    return np.sqrt(w * ur * u0 / 2 / cond)


def skin_depth(f, cond, ur=1):
    """Calculate skin depth.

    Args:
        f: frequency
        cond: conductivity
        ur: relative permeability

    Returns:
        np.ndarray: skin depth

    """

    return 1 / sqrt(pi * ur * u0 * cond * f)


def conductivity_rough(f, cond, roughness, ur=1, model='groiss'):
    """Calculate the effective conductivity of a rough metal.

    Using the Hammerstad-Bekkadal ('HB') or Groiss ('Groiss') model.

    Args:
        f: frequency
        cond: conductivity
        roughness: rms surface roughness
        ur: relative permeability
        model: roughness model, 'Groiss' or 'HB'

    Returns:
        np.ndarray: effective conductivity

    """

    ds = skin_depth(f, cond, ur=ur)

    if model.lower() == 'groiss':
        keff = 1 + np.exp(-(ds / 2 / roughness) ** 1.6)
    elif model.lower() == 'hb':
        keff = 1 + 2 / pi * arctan(1.4 * (roughness / ds) ** 2)
    else:
        print("Model not recognized.")
        raise ValueError

    return cond / keff ** 2


# WAVEGUIDE WITH DIELECTRIC SECTION -------------------------------------- ###

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


# HELPER FUNCTIONS ------------------------------------------------------- ###

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
