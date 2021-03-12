Waveguide
=========

*Calculate the properties of rectangular waveguides*

Example: Simple Waveguide
-------------------------

WR-90 waveguide:
```python
import numpy as np 
import scipy.constants as sc
import matplotlib.pyplot as plt
import waveguide as wg

# WR-90
a, b = 0.9*sc.inch, 0.45*sc.inch

# Conductivity, S/m
cond = 2e7

# Frequency sweep
freq = np.linspace(7, 13, 100) * sc.giga
```

Phase constant:
```python
beta = wg.phase_constant(freq, a, b, cond=cond)

plt.figure()
plt.plot(freq/1e9, beta)
plt.ylabel(r"Phase constant, $\beta$ (rad/m)")
plt.xlabel("Frequency (GHz)")
plt.xlim([7, 13])
```
<img src="https://raw.githubusercontent.com/garrettj403/Waveguide/main/examples/results/simple-waveguide-phase-constant.png" width="500">
Attenuation constant:
```python
alpha = wg.attenuation_constant(freq, a, b, cond=cond)

plt.figure()
plt.plot(freq/1e9, alpha)
plt.ylabel(r"Attenuation constant, $\beta$ (Np/m)")
plt.xlabel("Frequency (GHz)")
plt.xlim([7, 13])
```
<img src="https://raw.githubusercontent.com/garrettj403/Waveguide/main/examples/results/simple-waveguide-attenuation-constant.png" width="500">
