# -*- coding: utf-8 -*-
"""8.1, worked example 1: Check the outgoing point Green response away from its source.

Companion topic: Helmholtz and wavelength.
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
source_amplitude = 1.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
r = np.linspace(0.5, 5, 4001)
k = 2 * np.pi / wavelength
dr = r[1] - r[0]

# %% NUMERICAL CALCULATION
green = source_amplitude * np.exp(1j * k * r) / (4 * np.pi * r)
first = (green[2:] - green[:-2]) / (2 * dr)
second = (green[2:] - 2 * green[1:-1] + green[:-2]) / dr**2
residual = second + 2 * first / r[1:-1] + k**2 * green[1:-1]
relative_residual = abs(residual) / abs(k**2 * green[1:-1])

# %% CHECK AND INTERPRET
assert np.max(relative_residual) < 0.001
print("The homogeneous equation holds away from the singular point source.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(r, green.real)
    axes[0].set(xlabel='Source distance (um)', ylabel='Re G (inverse um)')
    axes[1].plot(r[1:-1], relative_residual)
    axes[1].set(xlabel='Source distance (um)', ylabel='Normalized Helmholtz residual')

    # %% NUMERICAL SUMMARY
    print('8.1 - Check the outgoing point Green response away from its source')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
