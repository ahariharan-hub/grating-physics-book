# -*- coding: utf-8 -*-
"""2.4, worked example 3: Compare uniform transverse spacing with nonuniform angle spacing.

Companion topic: Check the spacing.
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
period = 4.5
plane_distance = 5.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-4, 5)
alpha = orders * wavelength / period

# %% NUMERICAL CALCULATION
angles = np.arcsin(alpha)
plane_x = plane_distance * np.tan(angles)
alpha_steps = np.diff(alpha)
angle_steps = np.diff(np.rad2deg(angles))

# %% CHECK AND INTERPRET
assert np.allclose(alpha_steps, wavelength / period)
assert np.ptp(angle_steps) > 1
print("Equal order steps do not mean equal angular or flat-detector spacing.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(orders, np.rad2deg(angles), 'o-')
    axes[0].set(xlabel='Order m', ylabel='Angle (deg)')
    axes[1].plot(orders, plane_x, 'o-')
    axes[1].set(xlabel='Order m', ylabel='Flat-detector position (um)')

    # %% NUMERICAL SUMMARY
    print('2.4 - Compare uniform transverse spacing with nonuniform angle spacing')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
