# -*- coding: utf-8 -*-
"""1.2, worked example 4: Estimate local directions from a paraxial Gaussian phase.

Companion topic: What does a wavevector arrow tell us?.
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
waist = 4.0
distance = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-8, 8, 801)
k = 2 * np.pi / wavelength
zR = np.pi * waist**2 / wavelength
curvature = distance * (1 + (zR / distance)**2)

# %% NUMERICAL CALCULATION
phase = k * distance + k * x**2 / (2 * curvature) - np.arctan(distance / zR)
numerical_kx = np.gradient(np.unwrap(np.angle(np.exp(1j * phase))), x)
predicted_kx = k * x / curvature

# %% CHECK AND INTERPRET
assert np.allclose(numerical_kx[1:-1], predicted_kx[1:-1], atol=1e-10)
print("Away from its waist a finite Gaussian beam has curved, not planar, phase fronts.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, numerical_kx / k, label='Numerical transverse phase gradient / k')
    ax.plot(x, predicted_kx / k, '--', label='Paraxial prediction x/R')
    ax.set(xlabel='x (um)', ylabel='Local paraxial direction')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.2 - Estimate local directions from a paraxial Gaussian phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
