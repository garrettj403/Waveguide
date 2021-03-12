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

Example: Alumina-Filled Waveguide
---------------------------------

WR-28 waveguide filled with 1 inch alumina slug
```python
import numpy as np 
import scipy.constants as sc
import matplotlib.pyplot as plt 

import waveguide as wg

# WR-28
a, b = 0.28*sc.inch, 0.14*sc.inch

# Conductivity
cond = 1.8e7

# Frequency sweep
freq = np.linspace(22, 42, 401) * sc.giga

# Relativity permittivity
er = 9.3

# Alumina length
length = 1 * sc.inch

# Section lengths
total_length = 1.7 * sc.inch
length1 = (total_length - length) / 2
length2 = length
length3 = length1

# S-parameters
_, _, s21, _ = wg.dielectric_sparam(freq, a, b, er, 0, cond, length1, length2, length3)

fig, ax = plt.subplots()
ax.plot(freq/1e9, 20*np.log10(np.abs(s21)))
plt.ylabel(r"$S_{21}$ magnitude (dB)")
plt.xlabel("Frequency (GHz)")
plt.xlim([22, 42])
```

<img src="https://raw.githubusercontent.com/garrettj403/Waveguide/main/examples/results/alumina-filled-waveguide-sparam.png" width="500">
