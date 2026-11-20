# -*- coding: utf-8 -*-
"""5.1, worked example 4: Track the loss of propagating orders as groove-parallel incidence increases.

Companion topic: Move an order through the 3D boundary.
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
orders = np.arange(-3, 4)
show_plot = True

# %% SETUP: coordinates, sampling and constants
betas = np.linspace(0, 0.9, 301)
alpha = alpha_incident + orders * wavelength / period

# %% NUMERICAL CALCULATION
radicands = 1 - alpha[None, :]**2 - betas[:, None]**2
valid = radicands > 0
counts = valid.sum(axis=1)
boundary_radius = np.sqrt(1 - betas**2)

# %% CHECK AND INTERPRET
assert np.all(np.diff(counts) <= 0)
print("The available transverse disk narrows when the conserved groove component grows.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].step(betas, counts, where='mid')
    axes[0].set(xlabel='Incident beta', ylabel='Propagating retained candidates')
    axes[1].plot(betas, boundary_radius)
    axes[1].set(xlabel='Incident beta', ylabel='Allowed absolute alpha bound')

    # %% NUMERICAL SUMMARY
    print('5.1 - Track the loss of propagating orders as groove-parallel incidence increases')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
