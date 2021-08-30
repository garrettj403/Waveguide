"""Propagation in rectangular waveguides."""

import numpy as np

from numpy import pi, sqrt, arctan
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as u0
from scipy.constants import m_e, e, mil


# PROPAGATION CONSTANT (GAMMA = ALPHA + j * BETA) ------------------------- ###

def propagation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate propagation constant (complex value).

    Typically represented by: gamma

    Args:
        f: frequency, in units [Hz]
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls, in units [S/m]
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: propagation constant (complex value)

    """

    if b is None:
        b = a / 2
    
    k = wavenumber(f, er, ur)
    kc = cutoff_wavenumber(a, b, m, n)
    alpha_c = conductor_loss(f, cond, a, b, er=er, ur=ur)

    return 1j * sqrt(k ** 2 - kc ** 2) + alpha_c


def attenuation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate attenuation constant (real component of propagation constant).

    Includes conductor loss and dielectric loss.

    Typically represented by: alpha

    Args:
        f: frequency, in units [Hz]
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls, in units [S/m]
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: attenuation constant, in units [Np/m]

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n).real


def phase_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate phase constant (imaginary component of propagation constant).

    Typically represented by: beta

    Args:
        f: frequency, in units [Hz]
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls, in units [S/m]
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: phase constant, in units [rad/m]

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n).imag


# PROPAGATION PROPERTIES -------------------------------------------------- ###

def wavelength(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate guided wavelength.

    Typically represented by: lambda

    Args:
        f: frequency, in units [Hz]
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        cond: conductivity of waveguide walls, in units [S/m]
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: wavelength, in units [m]

    """

    beta = phase_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n)

    return 2 * pi / beta


def intrinsic_impedance(er=1, ur=1):
    """Calculate intrinsic impedance of dielectric material (or vacuum).

    Typically represented by: eta

    Args:
        er: relative permittivity
        ur: relative permeability

    Returns:
        float: intrinsic impedance, in units [ohms]

    """

    return sqrt((ur * u0) / (er * e0))


def wavenumber(f, er=1, ur=1):
    """Calculate freespace wavenumber of dielectric material (or vacuum).

    Typically represented by: k

    Args:
        f: frequency, in units [Hz]
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: freespace wavenumber, in units [1/m]

    """

    return 2 * pi * f * sqrt(ur * u0 * er * e0)


def cutoff_wavenumber(a, b=None, m=1, n=0):
    """Calculate cutoff wavenumber of TEmn or TMmn waveguide mode.

    Args:
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        m: mode number m
        n: mode number n

    Returns:
        float: cutoff wavenumber, in units [1/m]

    """

    # Assume standard dimensions if b is not provided
    if b is None:
        b = a / 2

    return sqrt((m * pi / a) ** 2 + (n * pi / b) ** 2)


def cutoff_frequency(a, b=None, er=1, ur=1, m=1, n=0):
    """Calculate cutoff frequency of TEmn or TMmn waveguide mode.

    Args:
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        m: mode number m
        n: mode number n

    Returns:
        float: cutoff frequency, in units [Hz]

    """

    kc = cutoff_wavenumber(a, b, m, n)

    return c0 / (2 * pi * sqrt(er * ur)) * kc


def impedance(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0, mode='TE'):
    """Calculate characteristic impedance of TEmn or TMmn waveguide mode.

    Args:
        f: frequency, in units [Hz]
        a: broad waveguide dimension, in units [m]
        b: narrow waveguide dimension, in units [m]
        er: relative permittivity
        ur: relative permeability
        cond: conductivity, in units [S/m]
        m: waveguide mode m
        n: waveguide mode n
        mode: mode, either "TE" or "TM"

    Returns:
        np.ndarray: characteristic impedance, in units [ohms]

    """

    k = wavenumber(f, er=er, ur=ur)
    eta = intrinsic_impedance(er=er, ur=ur)
    beta = phase_constant(f, a, b=b, er=er, ur=ur, cond=cond, m=m, n=n)

    if mode.lower() == 'te':
        return k * eta / beta
    elif mode.lower() == 'tm':
        return beta * eta / k
    else:
        print("Mode must be either TE or TM")
        raise ValueError


# LOSS -------------------------------------------------------------------- ###

def dielectric_loss(f, a, b=None, er=1, ur=1, m=1, n=0):
    """Calculate dielectric loss.

    Args:
        f: frequency, in units [Hz]
        a: broad dimension of waveguide, in units [m]
        b: narrow dimension of waveguide, in units [m]
        er: relative permittivity
        ur: relative permeability
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: dielectric loss, in units [Np/m]

    """

    return propagation_constant(f, a, b=b, er=er, ur=ur, cond=None, m=m, n=n).real


def conductor_loss(f, cond, a, b, er=1, ur=1):
    """Calculate loss due to conduction in waveguide walls.

    Only for TE10 mode.

    Args:
        f: frequency, in units [Hz]
        cond: conductivity of waveguide walls (~5.8e7 for copper), in units [S/m]
        a: broad waveguide dimension, in units [m]
        b: narrow waveguide dimension, in units [m]
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: conductor loss, in units [Np/m]

    """

    if cond is None:
        return np.zeros_like(f)

    # Propagation properties
    k = np.real(wavenumber(f, er=er.real, ur=ur))
    beta = np.imag(propagation_constant(f, a, b, er=er, ur=ur, m=1, n=0))
    eta = np.real(intrinsic_impedance(er=er, ur=ur))

    # Surface resistance
    rs = surface_resistance(f, cond, ur=ur.real)

    # Conductor loss (Eqn. 3.96 in Pozar)
    return rs / (a**3 * b * beta * k * eta) * (2 * b * pi**2 + a**3 * k**2)

    # # Conductor loss (Maxwell 1947)
    # lambda_c = 2 * a
    # lambda_0 = 2 * np.pi / k
    # term1 = 1 / 2 / b
    # term2 = np.sqrt(1 - (lambda_0 / lambda_c) ** 2)
    # term3 = np.sqrt(4 * np.pi / (lambda_0 * u0 * c0 * cond))
    # term4 = 1 + 2 * b / a * (lambda_0 / lambda_c) ** 2
    # return term1 / term2 * term3 * term4


# CONDUCTIVITY / SURFACE RESISTANCE --------------------------------------- ###

def surface_resistance(f, cond, ur=1):
    """Calculate surface resistance.

    Args:
        f: frequency, in units [Hz]
        cond: conductivity, in units [S/m]
        ur: relative permeability

    Returns:
        np.ndarray: surface resistance, in units [ohms/sq]

    """

    return np.sqrt(2 * pi * f * ur * u0 / 2 / cond)


def skin_depth(f, cond, ur=1):
    """Calculate skin depth.

    Args:
        f: frequency, in units [Hz]
        cond: conductivity, in units [S/m]
        ur: relative permeability

    Returns:
        np.ndarray: skin depth, in units [m]

    """

    return 1 / sqrt(pi * ur * u0 * cond * f)


def conductivity_rough(f, cond, roughness, ur=1, model='groiss'):
    """Calculate the effective conductivity of a rough metal.

    Using the Hammerstad-Bekkadal ('HB') or Groiss ('Groiss') model.

    Args:
        f: frequency, in units [Hz]
        cond: conductivity, in units [S/m]
        roughness: rms surface roughness, in units [m]
        ur: relative permeability
        model: roughness model, 'Groiss' or 'HB'

    Returns:
        np.ndarray: effective conductivity, in units [S/m]

    """

    ds = skin_depth(f, cond, ur=ur)

    if model.lower() == 'groiss':
        keff = 1 + np.exp(-(ds / 2 / roughness) ** 1.6)
    elif model.lower() == 'hb':
        keff = 1 + 2 / pi * arctan(1.4 * (roughness / ds) ** 2)
    else:
        print("Model not recognized.")
        raise ValueError

    return cond / (keff ** 2)


def effective_conductivity(f, alpha_c, a, b, er=1, ur=1):
    """Calculate effective conductivity from attenuation constant.

    Args:
        f: frequency, in units [Hz]
        alpha_c: conductor attenuation constant, in units [Np/m]
        a: waveguide dimension a, in units [m]
        b: waveguide dimension b, in units [m]
        er: relative permittivity
        ur: relative permeability

    Returns:
        effective conductivity, in units [S/m]

    """

    # Propagation properties
    k = np.real(wavenumber(f, er=er.real, ur=ur.real))
    beta = np.imag(propagation_constant(f, a, b, er=er, ur=ur, m=1, n=0))
    eta = np.real(intrinsic_impedance(er=er.real, ur=ur.real))

    # Surface resistance
    rs = alpha_c * (a**3 * b * beta * k * eta) / (2 * b * pi**2 + a**3 * k**2)

    # Effective conductivity
    return 2 * pi * f * ur * u0 / 2 / rs**2


def conductivity_ase(freq, fermi_speed, e_density, beta=1.5, mu_r=1):
    """Calculate the effective conductivity in the regime of the 
    anomalous skin effect.

    Args:
        freq (float): frequency in [Hz]
        fermi_speed (float): Fermi speed in [m/s]
        e_density (float): electron density [m-3]
        mu_r (float): relative permeability

    Returns:
        float: effective conductivity, in units [S/m]

    """

    return (beta ** 2 * e_density ** 2 * e ** 4 / 
        (pi * mu_r * u0 * m_e ** 2 * fermi_speed ** 2 * freq)) ** (1 / 3)


def conductivity_4k(freq, fermi_speed, e_density, beta=1.5, mu_r=1):
    """Calculate the effective conductivity at 4K, assuming that you are in 
    the regime of the anomalous skin effect.

    Args:
        freq (float): frequency in [Hz]
        fermi_speed (float): Fermi speed in [m/s]
        e_density (float): electron density [m-3]
        mu_r (float): relative permeability

    Returns:
        float: effective conductivity, in units [S/m]

    """

    return conductivity_ase(freq, fermi_speed, e_density, beta, mu_r)

