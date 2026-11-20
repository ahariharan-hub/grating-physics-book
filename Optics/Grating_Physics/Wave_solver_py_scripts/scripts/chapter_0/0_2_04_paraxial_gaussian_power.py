# -*- coding: utf-8 -*-
"""0.2, worked example 4: Implement Gaussian spreading and check transverse power.

Companion topic: Gaussian families and the paraxial approximation.
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
from scipy.integrate import trapezoid

# %% INPUT PARAMETERS
wavelength = 1.0
waist = 4.0
distances = np.array([0.0, 20.0, 60.0])
show_plot = True

# %% SETUP: coordinates, sampling and constants
rayleigh_range = np.pi * waist**2 / wavelength
widths = waist * np.sqrt(1 + (distances / rayleigh_range)**2)
r = np.linspace(0.0, 6 * widths.max(), 2001)

# %% NUMERICAL CALCULATION
intensities = (waist / widths[:, None])**2 * np.exp(-2 * (r / widths[:, None])**2)
powers = trapezoid(2 * np.pi * r * intensities, r, axis=1)
expected_power = np.pi * waist**2 / 2

# %% CHECK AND INTERPRET
assert np.allclose(powers, expected_power, rtol=2e-5)
print('Integrated transverse powers:', powers)
print('This is a paraxial Gaussian model, not an exact Helmholtz family.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for z, intensity in zip(distances, intensities):
        ax.plot(r, intensity, label=f'z={z:g} um')
    ax.set(xlabel='Radius (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.2 - Implement Gaussian spreading and check transverse power')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
