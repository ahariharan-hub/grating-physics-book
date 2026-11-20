# -*- coding: utf-8 -*-
"""2.2, worked example 1: Evaluate incident complex phase at neighbouring openings.

Companion topic: Illuminate the surface straight on and tilted.
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
period = 2.0
incidence_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
positions = np.arange(-3, 4) * period
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
phase = kx * positions
incident = np.exp(1j * phase)
ratios = incident[1:] / incident[:-1]

# %% CHECK AND INTERPRET
assert np.allclose(ratios, np.exp(1j * kx * period))
print("Neighbour-to-neighbour complex translation:", ratios[0])

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(positions, phase / np.pi, 'o-')
    axes[0].set(xlabel='Opening position (um)', ylabel='Unwrapped phase / pi')
    axes[1].plot(incident.real, incident.imag, 'o')
    axes[1].set(xlabel='Real incident field', ylabel='Imaginary incident field', aspect='equal')

    # %% NUMERICAL SUMMARY
    print('2.2 - Evaluate incident complex phase at neighbouring openings')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
