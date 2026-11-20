# -*- coding: utf-8 -*-
"""2.4, worked example 1: Convert one set of orders between k, frequency and direction cosine.

Companion topic: One order on three coordinate scales.
Lengths are micrometres. Wavelength is in the medium. Time convention: exp(-i omega t).
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
incidence_degrees = 0.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-2, 3)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
kx = k * np.sin(np.deg2rad(incidence_degrees)) + orders * 2 * np.pi / period
fx = kx / (2 * np.pi)
alpha = kx / k

# %% CHECK AND INTERPRET
assert np.allclose(alpha, wavelength * fx)
assert np.allclose(np.diff(kx), 2 * np.pi / period)
print("Rows m, kx, fx, alpha:", np.column_stack((orders, kx, fx, alpha)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, values, label in zip(axes, [kx, fx, alpha], ['kx (rad/um)', 'fx (cycles/um)', 'Direction cosine alpha']):
        ax.plot(orders, values, 'o-')
        ax.set(xlabel='Order m', ylabel=label)

    # %% NUMERICAL SUMMARY
    print('2.4 - Convert one set of orders between k, frequency and direction cosine')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
