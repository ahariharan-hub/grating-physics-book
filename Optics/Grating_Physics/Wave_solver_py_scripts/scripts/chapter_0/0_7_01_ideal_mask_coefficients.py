# -*- coding: utf-8 -*-
"""0.7, worked example 1: Integrate ideal binary-mask Fourier coefficients.

Companion topic: From the prescribed mask to common-plane coefficients.
Dimensionless unit-cell coordinate x/d. A prescribed ideal mask, not material scattering.
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
clear_fraction = 0.4
orders = np.arange(-6, 7)
samples = 20000
show_plot = True

# %% SETUP: coordinates, sampling and constants
fractional_x = -0.5 + (np.arange(samples) + 0.5) / samples

# %% NUMERICAL CALCULATION
mask = (abs(fractional_x) < clear_fraction / 2).astype(float)
coefficients = np.mean(mask[:, None] * np.exp(-2j * np.pi * fractional_x[:, None] * orders), axis=0)
analytic = clear_fraction * np.sinc(orders * clear_fraction)

# %% CHECK AND INTERPRET
assert np.max(abs(coefficients - analytic)) < 1e-7
print('The unit-cell mean is t0 =', coefficients[orders == 0][0])

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders, coefficients.real, 'o', label='Midpoint integration')
    ax.plot(orders, analytic, '--', label='D sinc(m D)')
    ax.set(xlabel='Order m', ylabel='Real coefficient')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.7 - Integrate ideal binary-mask Fourier coefficients')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
