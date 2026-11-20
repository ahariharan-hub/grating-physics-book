# -*- coding: utf-8 -*-
"""2.3, worked example 1: Enumerate every retained order before classifying it.

Companion topic: Meet the diffraction orders.
Medium wavelength and lengths um. Outgoing kz=-q toward -z, with Im(q)>=0.
Run the entire file in Python or Spyder, or execute its # %% sections in order.
Change INPUT PARAMETERS to investigate the result. No figures are saved automatically.
"""
from pathlib import Path
import sys
import numpy as np

# Locate only the plotting/path helper; the numerical implementation is below.
_scripts_directory = str(Path(__file__).resolve().parents[1])
if _scripts_directory not in sys.path:
    sys.path.insert(0, _scripts_directory)
from _common import pyplot, finish

# %% INPUT PARAMETERS
wavelength = 1.0
period = 2.5
incidence_degrees = 20.0
orders = np.arange(-5, 6)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
incident_kx = k * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
kx = incident_kx + 2 * np.pi * orders / period
q = np.sqrt((k**2 - kx**2).astype(complex))
propagating = abs(kx) < k
grazing = np.isclose(abs(kx), k)
classification = np.where(grazing, 'grazing', np.where(propagating, 'propagating', 'evanescent'))

# %% CHECK AND INTERPRET
assert np.allclose(kx**2 + q**2, k**2)
print(list(zip(orders, classification)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders, q.real / k, 'o-', label='Real longitudinal q/k')
    ax.plot(orders, q.imag / k, 's-', label='Imaginary longitudinal q/k')
    ax.set(xlabel='Order m', ylabel='Longitudinal component')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.3 - Enumerate every retained order before classifying it')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
