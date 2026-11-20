# -*- coding: utf-8 -*-
"""4.1, worked example 4: Propagate a boundary containing both real and imaginary longitudinal components.

Companion topic: Keep an order that has no real outgoing direction.
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
wavelength = 1.0
period = 3.0
orders = np.array([0, 1, 4])
coefficients = np.array([1.0, 0.4, 0.3j])
distance = 0.8
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
x = np.arange(512) * period / 512
kx = orders * 2 * np.pi / period

# %% NUMERICAL CALCULATION
q = np.sqrt((k**2 - kx**2).astype(complex))
basis = np.exp(1j * kx[:, None] * x)
boundary = coefficients @ basis
advanced_coefficients = coefficients * np.exp(1j * q * distance)
propagated = advanced_coefficients @ basis

# %% CHECK AND INTERPRET
assert np.allclose(abs(advanced_coefficients[q.imag == 0]), abs(coefficients[q.imag == 0]))
assert abs(advanced_coefficients[-1]) < abs(coefficients[-1])
print("Evanescent content is present at the boundary and decays during propagation.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, abs(boundary)**2, label='Boundary intensity')
    ax.plot(x, abs(propagated)**2, label='Propagated intensity')
    ax.set(xlabel='x over one period (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('4.1 - Propagate a boundary containing both real and imaginary longitudinal components')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
