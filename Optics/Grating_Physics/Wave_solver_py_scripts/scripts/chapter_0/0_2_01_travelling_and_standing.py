# -*- coding: utf-8 -*-
"""0.2, worked example 1: Build a standing wave from opposite travelling waves.

Companion topic: Plane waves and standing waves.
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
time_fraction = 0.2
show_plot = True

# %% SETUP: coordinates, sampling and constants
z = np.linspace(-4.0, 4.0, 801)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
forward = 0.5 * np.exp(1j * k * z)
backward = 0.5 * np.exp(-1j * k * z)
standing = forward + backward
real_field = np.real(standing * np.exp(-2j * np.pi * time_fraction))

# %% CHECK AND INTERPRET
assert np.allclose(standing, np.cos(k * z))
print('The spatial nodes are stationary; the real field oscillates in time.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(z, forward.real, '--', label='Forward component')
    ax.plot(z, backward.real, ':', label='Backward component')
    ax.plot(z, real_field, label='Standing real field')
    ax.set(xlabel='z (um)', ylabel='Scalar field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.2 - Build a standing wave from opposite travelling waves')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
