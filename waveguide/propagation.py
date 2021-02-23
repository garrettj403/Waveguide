"""Propagation properties."""

import numpy as np
from numpy import pi, sqrt, arctan
from scipy.constants import c as c0
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as u0


# PROPAGATION CONSTANT (GAMMA = ALPHA + j * BETA) ------------------------- ###

def propagation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate propagation constant (complex value).

    Typically represented by: gamma

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
        np.ndarray: propagation constant (complex value)

    """

    k = wavenumber(f, er, ur)
    kc = cutoff_wavenumber(a, b, m, n)
    alpha_c = conductor_loss(f, cond, a, b, er=er, ur=ur)

    return 1j * sqrt(k ** 2 - kc ** 2) + alpha_c


def attenuation_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate attenuation constant (real component of propagation constant).

    Includes conductor loss and dielectric loss.

    Typically represented by: alpha

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


def phase_constant(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate phase constant (imaginary component of propagation constant).

    Typically represented by: beta

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


# PROPAGATION PROPERTIES -------------------------------------------------- ###

def wavelength(f, a, b=None, er=1, ur=1, cond=None, m=1, n=0):
    """Calculate guided wavelength.

    Typically represented by: lambda

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


def intrinsic_impedance(er=1, ur=1):
    """Calculate intrinsic impedance of dielectric material (or vacuum).

    Typically represented by: eta

    Args:
        er: relative permittivity
        ur: relative permeability

    Returns:
        float: intrinsic impedance

    """

    return sqrt((ur * u0) / (er * e0))


def wavenumber(f, er=1, ur=1):
    """Calculate freespace wavenumber of dielectric material (or vacuum).

    Typically represented by: k

    Args:
        f: frequency
        er: relative permittivity
        ur: relative permeability

    Returns:
        np.ndarray: freespace wavenumber

    """

    return 2 * pi * f * sqrt(ur * u0 * er * e0)


def cutoff_wavenumber(a, b=None, m=1, n=0):
    """Calculate cutoff wavenumber of TEmn or TMmn waveguide mode.

    Args:
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        m: mode number m
        n: mode number n

    Returns:
        float: cutoff wavenumber

    """

    # Assume standard dimensions if b is not provided
    if b is None:
        b = a / 2

    return sqrt((m * pi / a) ** 2 + (n * pi / b) ** 2)


def cutoff_frequency(a, b=None, er=1, ur=1, m=1, n=0):
    """Calculate cutoff frequency of TEmn or TMmn waveguide mode.

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
    """Calculate characteristic impedance of TEmn or TMmn waveguide mode.

    Args:
        f: frequency
        a: broad waveguide dimension
        b: narrow waveguide dimension
        er: relative permittivity
        ur: relative permeability
        cond: conductivity
        m: waveguide mode m
        n: waveguide mode n
        mode: mode, either "TE" or "TM"

    Returns:
        np.ndarray: characteristic impedance

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
        f: frequency
        a: broad dimension of waveguide
        b: narrow dimension of waveguide
        er: relative permittivity
        ur: relative permeability
        m: mode number m
        n: mode number n

    Returns:
        np.ndarray: dielectric loss in Np/m

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


# CONDUCTIVITY / SURFACE RESISTANCE --------------------------------------- ###

def surface_resistance(f, cond, ur=1):
    """Calculate surface resistance.

    Args:
        f: frequency
        cond: conductivity
        ur: relative permeability

    Returns:
        np.ndarray: surface resistance

    """

    return np.sqrt(2 * pi * f * ur * u0 / 2 / cond)


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

    return cond / (keff ** 2)


def effective_conductivity(f, alpha_c, a, b, er=1, ur=1):

    # Propagation properties
    k = np.real(wavenumber(f, er=er.real, ur=ur.real))
    beta = np.imag(propagation_constant(f, a, b, er=er, ur=ur, m=1, n=0))
    eta = np.real(intrinsic_impedance(er=er.real, ur=ur.real))

    # Surface resistance
    rs = alpha_c * (a**3 * b * beta * k * eta) / (2 * b * pi**2 + a**3 * k**2)

    # Effective conductivity
    return 2 * pi * f * ur * u0 / 2 / rs**2
