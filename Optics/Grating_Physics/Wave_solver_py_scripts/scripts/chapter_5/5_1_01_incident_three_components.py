# -*- coding: utf-8 -*-
"""5.1, worked example 1: Complete an incident wavevector with a groove-parallel component.

Companion topic: Let the incident wave travel along the grooves.
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
alpha_incident = 0.3
beta_incident = 0.4
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
assert alpha_incident**2 + beta_incident**2 < 1
gamma_incident = -np.sqrt(1 - alpha_incident**2 - beta_incident**2)
vector = k * np.array([alpha_incident, beta_incident, gamma_incident])

# %% CHECK AND INTERPRET
assert np.isclose(np.dot(vector, vector), k**2)
print("Incident wavevector toward -z (rad/um):", vector)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure = plt.figure(figsize=(6, 5), layout='constrained')
    ax = figure.add_subplot(projection='3d')
    ax.quiver(0, 0, 0, *(vector / k))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='alpha', ylabel='beta', zlabel='gamma')
    ax.set_box_aspect((1, 1, 1))

    # %% NUMERICAL SUMMARY
    print('5.1 - Complete an incident wavevector with a groove-parallel component')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
