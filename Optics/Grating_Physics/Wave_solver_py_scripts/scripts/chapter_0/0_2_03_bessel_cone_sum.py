# -*- coding: utf-8 -*-
"""0.2, worked example 3: Construct a Bessel beam by angular quadrature.

Companion topic: Bessel beams as superpositions of plane waves.
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
from scipy.special import j0

# %% INPUT PARAMETERS
wavelength = 2.0
transverse_ratio = 0.6
angular_samples = 512
distance = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
rho = np.linspace(0.0, 8.0, 301)
azimuth = 2 * np.pi * np.arange(angular_samples) / angular_samples
k_transverse = k * transverse_ratio
k_longitudinal = np.sqrt(k**2 - k_transverse**2)

# %% NUMERICAL CALCULATION
components = np.exp(1j * k_transverse * rho[:, None] * np.cos(azimuth))
cone_sum = components.mean(axis=1) * np.exp(1j * k_longitudinal * distance)
analytic = j0(k_transverse * rho) * np.exp(1j * k_longitudinal * distance)

# %% CHECK AND INTERPRET
assert np.max(abs(cone_sum - analytic)) < 1e-10
print('Angular quadrature reproduces J0; this ideal beam has infinite extent.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(rho, cone_sum.real, label='Plane-wave sum')
    ax.plot(rho, analytic.real, '--', label='Bessel J0 reference')
    ax.set(xlabel='Radius (um)', ylabel='Real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.2 - Construct a Bessel beam by angular quadrature')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
