# -*- coding: utf-8 -*-
"""5.4, worked example 2: Apply a circular angular collection aperture to order candidates.

Companion topic: Put source points and grating orders on the same axes.
Vacuum wavelength um; source sigma held fixed as NA changes. No lens field transformation or image is computed.
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
vacuum_wavelength = 0.193
period = 0.6
source_sigma = 0.3
NA_values = np.linspace(0.2, 0.95, 151)
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-6, 7)

# %% NUMERICAL CALCULATION
sigma = source_sigma + orders[None, :] * vacuum_wavelength / (period * NA_values[:, None])
collected = abs(sigma) <= 1
counts = collected.sum(axis=1)
# For air and NA<1, collected directions are also propagating.
direction_cosines = sigma * NA_values[:, None]

# %% CHECK AND INTERPRET
assert np.all(abs(direction_cosines[collected]) < 1)
print("Collection geometry is calculated without assigning order powers or forming an image.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.step(NA_values, counts, where='mid')
    ax.set(xlabel='Objective NA in air', ylabel='Collected retained order candidates')

    # %% NUMERICAL SUMMARY
    print('5.4 - Apply a circular angular collection aperture to order candidates')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
