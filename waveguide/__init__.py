"""Rectangular waveguide properties."""

from .util import (db2np, np2db, db10, db20, rad2deg, deg2rad, linear10, 
                   linear20)
from .propagation import (propagation_constant, attenuation_constant,
                          phase_constant, wavelength, intrinsic_impedance,
                          wavenumber, cutoff_wavenumber, cutoff_frequency,
                          impedance, dielectric_loss, conductor_loss,
                          surface_resistance, skin_depth, conductivity_rough,
                          effective_conductivity, conductivity_4k)
from .cavity import (resonant_frequency, qfactor_dielectric, 
                     qfactor_conduction, qfactor_parallel, find_resonances, 
                     find_qfactor, q2surface_resistance, q2conductivity,
                     resonant_frequency2permittivity, deembed_qfactor,
                     q2loss_tangent, guess_resonance_order)
from .components import thin_iris
from .dielectric import dielectric_sparam

__author__ = "John Garrett"
__version__ = "0.0.3"
