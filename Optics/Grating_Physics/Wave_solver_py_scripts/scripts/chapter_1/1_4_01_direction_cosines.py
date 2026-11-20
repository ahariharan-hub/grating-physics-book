# -*- coding: utf-8 -*-
"""1.4, worked example 1: Construct a 3D wavevector from polar and azimuthal angles.

Companion topic: Three dimensions.
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
polar_degrees = 40.0
azimuth_degrees = 55.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
theta, phi = np.deg2rad([polar_degrees, azimuth_degrees])
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
vector = k * direction

# %% CHECK AND INTERPRET
assert np.isclose(np.dot(direction, direction), 1)
print("Wavevector (rad/um):", vector)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure = plt.figure(figsize=(6, 5), layout='constrained')
    ax = figure.add_subplot(projection='3d')
    ax.quiver(0, 0, 0, *direction)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='kx/k', ylabel='ky/k', zlabel='kz/k')
    ax.set_box_aspect((1, 1, 1))

    # %% NUMERICAL SUMMARY
    print('1.4 - Construct a 3D wavevector from polar and azimuthal angles')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
