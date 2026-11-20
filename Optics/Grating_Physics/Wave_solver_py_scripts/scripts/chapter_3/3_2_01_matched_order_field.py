# -*- coding: utf-8 -*-
"""3.2, worked example 1: Build a selected-order wave and check the surface seam.

Companion topic: Follow the wave into a selected order.
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
order = 1
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
x = np.linspace(0, period, 501)
kx_incident = k * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
kx_outgoing = kx_incident + order * 2 * np.pi / period
incident = np.exp(1j * kx_incident * x)
outgoing = np.exp(1j * kx_outgoing * x)
relative = outgoing / incident

# %% CHECK AND INTERPRET
assert np.isclose(relative[-1], relative[0])
print("Extra phase cycles:", (kx_outgoing - kx_incident) * period / (2 * np.pi))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, relative.real, label='Real relative phase factor')
    ax.plot(x, relative.imag, label='Imaginary relative phase factor')
    ax.set(xlabel='Displacement over one pitch (um)', ylabel='Relative complex field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('3.2 - Build a selected-order wave and check the surface seam')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
