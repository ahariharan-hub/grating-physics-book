# -*- coding: utf-8 -*-
"""7.2, worked example 3: Predict fringe displacement when the detector moves downstream.

Companion topic: Move the detector and change phase.
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
period = 4.0
orders = np.array([0, 1])
depths = np.array([0.0, 4.0, 8.0])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-8, 8, 1201)
k = 2 * np.pi / wavelength
kx = 2 * np.pi * orders / period
q = np.sqrt(k**2 - kx**2)

# %% NUMERICAL CALCULATION
delta_kx = kx[1] - kx[0]
delta_q = q[1] - q[0]
intensities = 2 + 2 * np.cos(delta_kx * x[None, :] + delta_q * depths[:, None])
shifts = -delta_q * depths / delta_kx
translated = 2 + 2 * np.cos(delta_kx * (x[None, :] - shifts[:, None]))

# %% CHECK AND INTERPRET
assert np.allclose(intensities, translated)
print("Predicted transverse displacements (um):", shifts)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for depth, intensity in zip(depths, intensities):
        ax.plot(x, intensity, label=f'Depth {depth:g} um')
    ax.set(xlabel='Detector x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.2 - Predict fringe displacement when the detector moves downstream')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
