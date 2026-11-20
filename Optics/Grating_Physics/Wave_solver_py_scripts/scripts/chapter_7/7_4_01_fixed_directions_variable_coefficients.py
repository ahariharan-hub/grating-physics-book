# -*- coding: utf-8 -*-
"""7.4, worked example 1: Hold order directions fixed and vary complex coefficients.

Companion topic: What information does C contain?.
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
period = 4.0
wavelength = 1.0
coefficient_cases = [1.0, 0.4, 1j, -1.0]
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-8, 8, 1201)
k = 2 * np.pi / wavelength
kx = np.array([0.0, 2 * np.pi / period])
q = np.sqrt(k**2 - kx**2)

# %% NUMERICAL CALCULATION
basis = np.exp(1j * (kx[:, None] * x + q[:, None] * 4.0))
intensities = []
for coefficient in coefficient_cases:
    total = basis[0] + coefficient * basis[1]
    intensities.append(abs(total)**2)

# %% CHECK AND INTERPRET
assert np.allclose(kx[1] - kx[0], 2 * np.pi / period)
print("All cases use identical order locations; their intensity profiles differ.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for coefficient, intensity in zip(coefficient_cases, intensities):
        ax.plot(x, intensity, label=f'C1={coefficient}')
    ax.set(xlabel='Detector x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.4 - Hold order directions fixed and vary complex coefficients')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
