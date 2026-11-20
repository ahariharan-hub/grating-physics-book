# -*- coding: utf-8 -*-
"""5.3, worked example 2: Classify a two-integer order lattice against the output propagation disk.

Companion topic: Two-dimensional diffraction orders.
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
pitch_x = 4.0
pitch_y = 5.0
incident_direction = np.array([0.3, 0.4])
show_plot = True

# %% SETUP: coordinates, sampling and constants
labels = np.array([(m, n) for m in range(-5, 6) for n in range(-5, 6)])
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
transverse = k * incident_direction + 2 * np.pi * labels / np.array([pitch_x, pitch_y])
q = np.sqrt((k**2 - np.sum(transverse**2, axis=1)).astype(complex))
propagating = q.real > 0

# %% CHECK AND INTERPRET
assert np.allclose(np.sum(transverse**2, axis=1) + q**2, k**2)
print("Propagating candidates:", labels[propagating])

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    angle = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(angle), np.sin(angle), color='black')
    ax.scatter(transverse[:, 0] / k, transverse[:, 1] / k, c=propagating, cmap='coolwarm')
    ax.set(xlabel='kx/k', ylabel='ky/k', aspect='equal', title='Allowed locations do not specify amplitudes')

    # %% NUMERICAL SUMMARY
    print('5.3 - Classify a two-integer order lattice against the output propagation disk')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
