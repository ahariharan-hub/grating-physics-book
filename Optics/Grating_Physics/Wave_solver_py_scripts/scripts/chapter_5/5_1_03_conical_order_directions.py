# -*- coding: utf-8 -*-
"""5.1, worked example 3: Construct conical outgoing directions and check their common cone angle.

Companion topic: Explore the actual outgoing directions.
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
period = 4.0
alpha_incident = 0.3
beta_incident = 0.4
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-4, 5)
alpha = alpha_incident + orders * wavelength / period
radicand = 1 - alpha**2 - beta_incident**2

# %% NUMERICAL CALCULATION
valid = radicand > 0
directions = np.column_stack([alpha[valid], np.full(valid.sum(), beta_incident), -np.sqrt(radicand[valid])])
cone_angles = np.rad2deg(np.arccos(directions[:, 1]))

# %% CHECK AND INTERPRET
assert np.allclose(np.sum(directions**2, axis=1), 1)
assert np.allclose(cone_angles, cone_angles[0])
print("Cone angle about the groove axis (deg):", cone_angles[0])

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure = plt.figure(figsize=(7, 6), layout='constrained')
    ax = figure.add_subplot(projection='3d')
    for order, direction in zip(orders[valid], directions):
        ax.quiver(0, 0, 0, *direction)
        ax.text(*direction, str(order))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 0), xlabel='alpha', ylabel='beta', zlabel='gamma')
    ax.set_box_aspect((1, 1, 1))

    # %% NUMERICAL SUMMARY
    print('5.1 - Construct conical outgoing directions and check their common cone angle')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
