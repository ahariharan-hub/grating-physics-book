# -*- coding: utf-8 -*-
"""0.2, worked example 2: Verify spherical and cylindrical radial solutions.

Companion topic: Outgoing spherical and cylindrical waves.
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
from scipy.special import hankel1

# %% INPUT PARAMETERS
wavelength = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
r = np.linspace(0.5, 8.0, 3001)
dr = r[1] - r[0]
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
spherical = np.exp(1j * k * r) / r
cylindrical = hankel1(0, k * r)
residuals = []
for field, radial_factor in [(spherical, 2.0), (cylindrical, 1.0)]:
    first = (field[2:] - field[:-2]) / (2 * dr)
    second = (field[2:] - 2 * field[1:-1] + field[:-2]) / dr**2
    residual = second + radial_factor * first / r[1:-1] + k**2 * field[1:-1]
    residuals.append(np.max(abs(residual)) / np.max(abs(k**2 * field[1:-1])))

# %% CHECK AND INTERPRET
assert max(residuals) < 0.001
print('Normalized finite-difference residuals:', residuals)
print('The source at r=0 is excluded from the homogeneous equation.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(r, abs(spherical), label='Spherical magnitude')
    ax.plot(r, abs(cylindrical), label='Cylindrical magnitude')
    ax.set(xlabel='Radius (um)', ylabel='Amplitude, separate normalizations')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.2 - Verify spherical and cylindrical radial solutions')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
