# -*- coding: utf-8 -*-
"""5.2, worked example 3: Compare 3D sphere projections and detector-plane intersections.

Companion topic: Compare a sphere projection with a plane.
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
alpha_incident = 0.3
beta_incident = 0.4
wavelength = 2.0
period = 4.0
radius = 4.0
plane_depth = 4.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-3, 4)
alpha = alpha_incident + orders * wavelength / period
radicand = 1 - alpha**2 - beta_incident**2

# %% NUMERICAL CALCULATION
valid = radicand > 0
transverse = np.column_stack([alpha[valid], np.full(valid.sum(), beta_incident)])
gamma = np.sqrt(radicand[valid])
sphere_xy = radius * transverse
plane_xy = plane_depth * transverse / gamma[:, None]

# %% CHECK AND INTERPRET
assert np.allclose(np.sum(sphere_xy**2, axis=1) + (radius * gamma)**2, radius**2)
print("A direction-coordinate projection is not a flat detector or a physical pupil mapping.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    ax.scatter(sphere_xy[:, 0], sphere_xy[:, 1], label='Reference sphere projection')
    ax.scatter(plane_xy[:, 0], plane_xy[:, 1], marker='x', label='Flat-plane intersection')
    ax.set(xlabel='x (um)', ylabel='y (um)', aspect='equal')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.2 - Compare 3D sphere projections and detector-plane intersections')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
