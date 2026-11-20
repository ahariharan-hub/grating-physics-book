# -*- coding: utf-8 -*-
"""0.1, worked example 1: Convert vacuum wavelength to medium wavelength.

Companion topic: Refractive index and the monochromatic field.
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
vacuum_wavelength = 0.6
refractive_indices = np.linspace(1.0, 2.0, 101)
speed_of_light = 0.299792458  # um/fs
show_plot = True

# %% SETUP: coordinates, sampling and constants
omega = 2 * np.pi * speed_of_light / vacuum_wavelength

# %% NUMERICAL CALCULATION
medium_speed = speed_of_light / refractive_indices
medium_wavelength = vacuum_wavelength / refractive_indices
k_from_frequency = omega / medium_speed
k_from_wavelength = 2 * np.pi / medium_wavelength

# %% CHECK AND INTERPRET
assert np.allclose(k_from_frequency, k_from_wavelength)
print('Frequency is fixed; wavelength and phase speed change with index.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(refractive_indices, medium_wavelength)
    axes[0].set(xlabel='Refractive index n', ylabel='Medium wavelength (um)')
    axes[1].plot(refractive_indices, k_from_frequency)
    axes[1].set(xlabel='Refractive index n', ylabel='Wavenumber (rad/um)')

    # %% NUMERICAL SUMMARY
    print('0.1 - Convert vacuum wavelength to medium wavelength')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
