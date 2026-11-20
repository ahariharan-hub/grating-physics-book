# -*- coding: utf-8 -*-
"""6.2, worked example 1: Sample a tilted order at three detector locations.

Companion topic: Compare a field snapshot with the detector image.
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
wavelength = 2.0
amplitude = 0.5
reference_amplitude = 1.0
time_fraction = 0.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
vector = k * np.array([0.5, 0.0, -np.sqrt(3) / 2])
points = np.array([[-1, 0, -1.5], [0, 0, -1.5], [1, 0, -1.5]])

# %% NUMERICAL CALCULATION
phasor = amplitude * np.exp(1j * (points @ vector))
instantaneous = np.real(phasor * np.exp(-2j * np.pi * time_fraction))
intensity = abs(phasor)**2 / reference_amplitude**2

# %% CHECK AND INTERPRET
assert np.allclose(intensity, (amplitude / reference_amplitude)**2)
print("Position, real field, relative intensity:", np.column_stack([points[:, 0], instantaneous, intensity]))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(points[:, 0], instantaneous, 'o-', label='Instantaneous field')
    ax.plot(points[:, 0], intensity, 's-', label='Relative intensity')
    ax.set(xlabel='Detector x (um)', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('6.2 - Sample a tilted order at three detector locations')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
