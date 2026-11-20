# -*- coding: utf-8 -*-
"""1.1, worked example 2: Compute travelling-wave snapshots in both directions.

Companion topic: Wavelength and propagation direction.
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
wavelength = 4.0
time_fraction = 0.25
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-8, 8, 801)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
right = np.cos(k * x - 2 * np.pi * time_fraction)
left = np.cos(-k * x - 2 * np.pi * time_fraction)
translated_initial = np.cos(k * (x - wavelength * time_fraction))

# %% CHECK AND INTERPRET
assert np.allclose(right, translated_initial)
print("Right-going displacement (um):", wavelength * time_fraction)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, right, label='Right going')
    ax.plot(x, left, '--', label='Left going')
    ax.set(xlabel='x (um)', ylabel='Real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.1 - Compute travelling-wave snapshots in both directions')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
