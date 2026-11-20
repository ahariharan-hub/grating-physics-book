# -*- coding: utf-8 -*-
"""1.3, worked example 2: Measure phase change along and across a wavevector.

Companion topic: Walk along or perpendicular to the wavevector.
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
angle_degrees = 45.0
step = 1.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
angle = np.deg2rad(angle_degrees)
direction = np.array([np.sin(angle), np.cos(angle)])

# %% NUMERICAL CALCULATION
vector = k * direction
displacements = np.array([step * direction, step * np.array([direction[1], -direction[0]])])
phase_changes = displacements @ vector
phasor_ratios = np.exp(1j * phase_changes)

# %% CHECK AND INTERPRET
assert np.allclose(phase_changes, [k * step, 0])
print("Along/across phase changes (rad):", phase_changes)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.bar(['Along k', 'Across k'], phase_changes)
    ax.set(ylabel='Phase change (rad)')

    # %% NUMERICAL SUMMARY
    print('1.3 - Measure phase change along and across a wavevector')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
