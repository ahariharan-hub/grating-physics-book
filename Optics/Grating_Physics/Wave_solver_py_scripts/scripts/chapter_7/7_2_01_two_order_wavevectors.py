# -*- coding: utf-8 -*-
"""7.2, worked example 1: Construct two grating-order vectors and verify their magnitudes.

Companion topic: Two orders and one optical field.
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
incidence_degrees = 20.0
orders = np.array([-1, 1])
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
incident_kx = k * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
kx = incident_kx + 2 * np.pi * orders / period
assert np.all(abs(kx) < k)
kz = -np.sqrt(k**2 - kx**2)
vectors = np.column_stack([kx, np.zeros(2), kz])
delta = vectors[1] - vectors[0]

# %% CHECK AND INTERPRET
assert np.allclose(np.linalg.norm(vectors, axis=1), k)
assert np.isclose(delta[0], 2 * np.pi * (orders[1] - orders[0]) / period)
print("Vector difference (rad/um):", delta)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure = plt.figure(figsize=(6, 5), layout='constrained')
    ax = figure.add_subplot(projection='3d')
    for order, vector in zip(orders, vectors / k):
        ax.quiver(0, 0, 0, *vector)
        ax.text(*vector, str(order))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 0), xlabel='kx/k', ylabel='ky/k', zlabel='kz/k')

    # %% NUMERICAL SUMMARY
    print('7.2 - Construct two grating-order vectors and verify their magnitudes')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
