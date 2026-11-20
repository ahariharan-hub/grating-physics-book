# -*- coding: utf-8 -*-
"""1.4, worked example 2: Evaluate a 3D field using matrix multiplication and explicit loops.

Companion topic: One number at each 3D position.
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
polar_degrees = 45.0
azimuth_degrees = 30.0
time_fraction = 0.2
show_plot = True

# %% SETUP: coordinates, sampling and constants
theta, phi = np.deg2rad([polar_degrees, azimuth_degrees])
k = 2 * np.pi / wavelength
points = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

# %% NUMERICAL CALCULATION
vector = k * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
phase_vectorized = points @ vector
phase_loop = np.array([sum(point[j] * vector[j] for j in range(3)) for point in points])
field = np.cos(phase_vectorized - 2 * np.pi * time_fraction)

# %% CHECK AND INTERPRET
assert np.allclose(phase_vectorized, phase_loop)
print("Point coordinates and real field:", np.column_stack((points, field)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.stem(np.arange(len(points)), field)
    ax.set(xlabel='Observation-point index', ylabel='Real scalar field')

    # %% NUMERICAL SUMMARY
    print('1.4 - Evaluate a 3D field using matrix multiplication and explicit loops')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
