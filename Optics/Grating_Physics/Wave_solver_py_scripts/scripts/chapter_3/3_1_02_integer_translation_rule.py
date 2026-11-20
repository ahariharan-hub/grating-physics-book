# -*- coding: utf-8 -*-
"""3.1, worked example 2: Recover allowed transverse components from discrete translation.

Companion topic: Connect phase change to diffraction orders.
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
incidence_degrees = 15.0
orders = np.arange(-3, 4)
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
incident_kx = k * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
outgoing_kx = incident_kx + orders * 2 * np.pi / period
relative_translation = np.exp(1j * (outgoing_kx - incident_kx) * period)

# %% CHECK AND INTERPRET
assert np.allclose(relative_translation, 1)
print("Discrete symmetry fixes transverse increments but not coefficient amplitudes.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.stem(orders, outgoing_kx / k)
    ax.set(xlabel='Integer order m', ylabel='Candidate outgoing kx/k')

    # %% NUMERICAL SUMMARY
    print('3.1 - Recover allowed transverse components from discrete translation')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
