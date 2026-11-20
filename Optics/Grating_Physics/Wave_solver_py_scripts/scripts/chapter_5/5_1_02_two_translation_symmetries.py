# -*- coding: utf-8 -*-
"""5.1, worked example 2: Test discrete x translation and continuous y translation.

Companion topic: What does the grating preserve?.
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
order = 1
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
displacements_y = np.linspace(-3, 3, 401)

# %% NUMERICAL CALCULATION
incident_kx = k * alpha_incident
incident_ky = k * beta_incident
outgoing_kx = incident_kx + 2 * np.pi * order / period
outgoing_ky = incident_ky
x_translation = np.exp(1j * (outgoing_kx - incident_kx) * period)
y_translations = np.exp(1j * (outgoing_ky - incident_ky) * displacements_y)
incorrect_y = np.exp(1j * 0.5 * displacements_y)

# %% CHECK AND INTERPRET
assert np.isclose(x_translation, 1)
assert np.allclose(y_translations, 1)
print("Continuous groove symmetry forces ky conservation for every displacement.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(displacements_y, abs(y_translations - 1), label='Conserved ky')
    ax.plot(displacements_y, abs(incorrect_y - 1), label='Incorrect changed ky')
    ax.set(xlabel='Groove-parallel displacement (um)', ylabel='Translation mismatch')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.1 - Test discrete x translation and continuous y translation')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
