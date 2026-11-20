# -*- coding: utf-8 -*-
"""5.4, worked example 1: Normalize diffraction wavevectors using objective numerical aperture.

Companion topic: Normalize the diffraction wavevectors.
Vacuum wavelength and lengths um; air exterior. Angular sigma coordinates are not themselves physical pupil positions.
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
vacuum_wavelength = 0.193
period = 0.6
NA = 0.75
source_sigma = 0.3
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-5, 6)
k0 = 2 * np.pi / vacuum_wavelength

# %% NUMERICAL CALCULATION
incident_kx = k0 * NA * source_sigma
kx = incident_kx + orders * 2 * np.pi / period
sigma = kx / (k0 * NA)
expected_sigma = source_sigma + orders * vacuum_wavelength / (period * NA)

# %% CHECK AND INTERPRET
assert np.allclose(sigma, expected_sigma)
print("The illumination position and grating orders share one sigma scale.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.scatter(sigma, np.zeros_like(sigma))
    ax.axvline(-1, color='gray', linestyle=':')
    ax.axvline(1, color='gray', linestyle=':')
    ax.set(xlabel='Angular sigma x', ylabel='Angular sigma y', title='One source point and its order row')

    # %% NUMERICAL SUMMARY
    print('5.4 - Normalize diffraction wavevectors using objective numerical aperture')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
