# -*- coding: utf-8 -*-
"""8.1, worked example 2: Calculate distance-dependent Green amplitude and phase.

Companion topic: What changes between two points?.
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
source_x = -1.0
source_s = -1.5
observation_s = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-5, 5, 1001)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
distance = np.sqrt((x - source_x)**2 + (observation_s - source_s)**2)
green = np.exp(1j * k * distance) / (4 * np.pi * distance)
unwrapped_path_phase = k * distance

# %% CHECK AND INTERPRET
assert np.allclose(abs(green) * 4 * np.pi * distance, 1)
print("Amplitude decreases as 1/R, while unwrapped path phase increases as kR.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, abs(green))
    axes[0].set(xlabel='Observation x (um)', ylabel='Green amplitude (inverse um)')
    axes[1].plot(x, unwrapped_path_phase / (2 * np.pi))
    axes[1].set(xlabel='Observation x (um)', ylabel='Unwrapped path cycles')

    # %% NUMERICAL SUMMARY
    print('8.1 - Calculate distance-dependent Green amplitude and phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
