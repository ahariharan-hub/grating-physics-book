# -*- coding: utf-8 -*-
"""5.3, worked example 3: Rotate direct and reciprocal lattices consistently.

Companion topic: Translate rescale and rotate the lattice.
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
direct = np.diag([4.0, 5.0])
rotation_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
angle = np.deg2rad(rotation_degrees)
rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
labels = np.array([(m, n) for m in range(-2, 3) for n in range(-2, 3)])

# %% NUMERICAL CALCULATION
reciprocal = 2 * np.pi * np.linalg.inv(direct).T
rotated_direct = rotation @ direct
rotated_reciprocal = 2 * np.pi * np.linalg.inv(rotated_direct).T
original_points = labels @ reciprocal.T
rotated_points = labels @ rotated_reciprocal.T

# %% CHECK AND INTERPRET
assert np.allclose(rotated_reciprocal, rotation @ reciprocal)
assert np.allclose(np.linalg.norm(original_points, axis=1), np.linalg.norm(rotated_points, axis=1))
print("A rigid rotation preserves reciprocal-vector lengths.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.scatter(original_points[:, 0], original_points[:, 1], label='Original')
    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], marker='x', label='Rotated')
    ax.set(xlabel='kx (rad/um)', ylabel='ky (rad/um)', aspect='equal')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.3 - Rotate direct and reciprocal lattices consistently')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
