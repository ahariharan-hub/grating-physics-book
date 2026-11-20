# -*- coding: utf-8 -*-
"""3.1, worked example 1: Calculate the incident phase difference across one pitch.

Companion topic: Compare two positions at the same time.
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
period = 2.0
incidence_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(0, period, 401)
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
phase = kx * x
field = np.exp(1j * phase)
translation = field[-1] / field[0]

# %% CHECK AND INTERPRET
assert np.isclose(translation, np.exp(1j * kx * period))
print("Unwrapped phase cycles over one pitch:", phase[-1] / (2 * np.pi))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, phase / (2 * np.pi))
    ax.set(xlabel='Surface displacement (um)', ylabel='Unwrapped incident phase cycles')

    # %% NUMERICAL SUMMARY
    print('3.1 - Calculate the incident phase difference across one pitch')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
