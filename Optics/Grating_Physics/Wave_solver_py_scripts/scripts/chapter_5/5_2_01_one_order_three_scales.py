# -*- coding: utf-8 -*-
"""5.2, worked example 1: Represent identical orders in k, spatial frequency and direction coordinates.

Companion topic: Direction cosines and Goodmans frequency convention.
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
orders = np.arange(-3, 4)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
alpha = alpha_incident + orders * wavelength / period
beta = np.full_like(alpha, beta_incident)
transverse_k = k * np.column_stack([alpha, beta])
frequencies = transverse_k / (2 * np.pi)
directions = transverse_k / k

# %% CHECK AND INTERPRET
assert np.allclose(directions, wavelength * frequencies)
print("Reciprocal spacing in fx:", 1 / period, "cycles/um")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, points, label in zip(axes, [transverse_k, frequencies, directions], ['k (rad/um)', 'f (cycles/um)', 'Direction cosines']):
        ax.scatter(points[:, 0], points[:, 1])
        ax.set(xlabel='Transverse x coordinate', ylabel='Transverse y coordinate', title=label, aspect='equal')

    # %% NUMERICAL SUMMARY
    print('5.2 - Represent identical orders in k, spatial frequency and direction coordinates')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
