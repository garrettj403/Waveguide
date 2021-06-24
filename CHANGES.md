v0.0.3 (Jun 24, 2021)
=====================

- Waveguide package:
	- Propagation module:
		- add conductivity_4k function to calculate conductivity at cryogenic temperatures using the anomalous skin effect model
	- Cavity module:
		- add ability to correct for S11 phase from iris or shunt susceptance
		- add function to estimate resonance order
	- Utility module:
		- add linear10, linear20 functions
	- Components module:
		- add thin_iris function to calculate the theoretical S-parameters of a thin iris
- Examples:
	- add simple waveguide example
	- add example of a WR-6.5 waveguide loaded with HDPE
	- add example comparing HFSS to theory
	- add example comparing conductor and dielectric loss
- Tests:
	- test against Scikit-RF
	- add example 4.14 in Gustrau, add test for lossy cavity

v0.0.2 (Mar 04, 2021)
=====================

- Waveguide package:
	- convert waveguide.py into a package
	- add q2loss_tangent
	- add functions to analyze Q-factor
	- add function to calculate resonant frequency of cavity
- Tests:
	- move test.py into directory
	- add test for simulated cavity
	- add data for testing cavity functions
	- add tests for utilities
	- add ability to plot tests
	- add tests for dielectric/conductor loss and effective conductivity
	- add function to calculate effective conductivity from alpha_c

v0.0.1 (Jan 26, 2021)
=====================

Initial release.
