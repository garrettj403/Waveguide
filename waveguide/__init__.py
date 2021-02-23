"""Rectangular waveguide properties."""

from .util import db2np, np2db, db10, db20, rad2deg, deg2rad
from .propagation import (propagation_constant, attenuation_constant,
                          phase_constant, wavelength, intrinsic_impedance,
                          wavenumber, cutoff_wavenumber, cutoff_frequency,
                          impedance, dielectric_loss, conductor_loss,
                          surface_resistance, skin_depth, conductivity_rough,
                          effective_conductivity)
from .cavity import resonant_frequency
from .dielectric import dielectric_sparam

__author__ = "John Garrett"
__version__ = "0.0.1-dev"
