# -*- coding: utf-8 -*-
"""2.4, worked example 4: Translate a direction-circle representation into scattering-vector coordinates.

Companion topic: The same orders on two surfaces.
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
wavelength = 1.0
period = 3.0
incidence_degrees = 20.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-6, 7)
incident = np.array([np.sin(np.deg2rad(incidence_degrees)), -np.cos(np.deg2rad(incidence_degrees))])

# %% NUMERICAL CALCULATION
alpha = incident[0] + orders * wavelength / period
valid = abs(alpha) < 1
directions = np.column_stack((alpha[valid], -np.sqrt(1 - alpha[valid]**2)))
scattering = directions - incident

# %% CHECK AND INTERPRET
assert np.allclose(np.sum(directions**2, axis=1), 1)
assert np.allclose(np.sum((scattering + incident)**2, axis=1), 1)
print("The two coordinate pictures differ by the incident-vector translation.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, points, title in zip(axes, [directions, scattering], ['Outgoing k/k', 'Change (k_out-k_in)/k']):
        ax.scatter(points[:, 0], points[:, 1])
        ax.set(xlabel='Transverse coordinate', ylabel='Normal coordinate', title=title, aspect='equal')

    # %% NUMERICAL SUMMARY
    print('2.4 - Translate a direction-circle representation into scattering-vector coordinates')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
