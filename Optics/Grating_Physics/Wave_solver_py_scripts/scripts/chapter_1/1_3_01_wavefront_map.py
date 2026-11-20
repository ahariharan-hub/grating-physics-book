# -*- coding: utf-8 -*-
"""1.3, worked example 1: Draw constant-phase lines and the perpendicular wavevector.

Companion topic: Wavefronts and wavevectors.
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
angle_degrees = 35.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-4, 4, 241)
z = np.linspace(0, 8, 241)
xx, zz = np.meshgrid(x, z)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
direction = np.array([np.sin(np.deg2rad(angle_degrees)), np.cos(np.deg2rad(angle_degrees))])
vector = k * direction
phase = vector[0] * xx + vector[1] * zz
tangent = np.array([direction[1], -direction[0]])

# %% CHECK AND INTERPRET
assert abs(np.dot(vector, tangent)) < 1e-12
print("Wavevector dotted with a wavefront tangent:", np.dot(vector, tangent))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.contour(xx, zz, phase, levels=np.arange(-20, 40) * np.pi)
    ax.quiver(0, 2, direction[0], direction[1], angles='xy', scale_units='xy', scale=0.5, color='red')
    ax.set(xlabel='x (um)', ylabel='z (um), +z convention', aspect='equal')

    # %% NUMERICAL SUMMARY
    print('1.3 - Draw constant-phase lines and the perpendicular wavevector')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
