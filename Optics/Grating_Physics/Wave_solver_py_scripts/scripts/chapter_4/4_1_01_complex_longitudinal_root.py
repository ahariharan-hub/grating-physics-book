# -*- coding: utf-8 -*-
"""4.1, worked example 1: Compute propagating and evanescent longitudinal components.

Companion topic: Complete the normal component.
Medium wavelength and lengths um. Downstream s=-z; U=exp(i kx x+i q s); positive-imaginary q decays.
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
wavelength = 2.0
period = 2.0
incidence_degrees = 30.0
orders = np.arange(-3, 4)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
incident_alpha = np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
alpha = incident_alpha + orders * wavelength / period
q = k * np.sqrt((1 - alpha**2).astype(complex))
classification = np.where(abs(q.imag) > 1e-12, 'evanescent', np.where(abs(q.real) < 1e-12, 'grazing', 'propagating'))

# %% CHECK AND INTERPRET
assert np.allclose((k * alpha)**2 + q**2, k**2)
assert np.all(q.imag >= 0)
print(list(zip(orders, classification)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders, q.real / k, 'o-', label='Real q/k')
    ax.plot(orders, q.imag / k, 's-', label='Imaginary q/k')
    ax.set(xlabel='Order m', ylabel='Longitudinal component')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('4.1 - Compute propagating and evanescent longitudinal components')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
