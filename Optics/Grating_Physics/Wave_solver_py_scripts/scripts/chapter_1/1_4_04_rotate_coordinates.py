# -*- coding: utf-8 -*-
"""1.4, worked example 4: Rotate the field geometry without changing its physical phase.

Companion topic: Wavevectors and coordinate frames.
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
rotation_degrees = 40.0
wavelength = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
angle = np.deg2rad(rotation_degrees)
points = np.array([[1, 0, 2], [0, 1, 1], [2, -1, 0]], dtype=float)
vector = 2 * np.pi / wavelength * np.array([0.3, 0.4, np.sqrt(0.75)])

# %% NUMERICAL CALCULATION
rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
rotated_vector = rotation @ vector
rotated_points = points @ rotation.T
old_phases = points @ vector
new_phases = rotated_points @ rotated_vector

# %% CHECK AND INTERPRET
assert np.allclose(old_phases, new_phases)
assert np.isclose(np.linalg.norm(vector), np.linalg.norm(rotated_vector))
print("k dot r is invariant when both vectors use the rotated frame.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(old_phases, 'o-', label='Original phases')
    ax.plot(new_phases, 'x--', label='Rotated phases')
    ax.set(xlabel='Observation-point index', ylabel='Spatial phase (rad)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.4 - Rotate the field geometry without changing its physical phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
