# -*- coding: utf-8 -*-
"""2.1, worked example 1: Resolve a single opening by direct far-field quadrature.

Companion topic: Start with one isolated slit.
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
wavelength = 1.0
width = 3.0
aperture_samples = 1600
show_plot = True

# %% SETUP: coordinates, sampling and constants
angles = np.linspace(-70, 70, 501)
xi = -width / 2 + (np.arange(aperture_samples) + 0.5) * width / aperture_samples
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
amplitude = np.exp(-1j * k * np.sin(np.deg2rad(angles))[:, None] * xi).mean(axis=1)
analytic = np.sinc(width * np.sin(np.deg2rad(angles)) / wavelength)

# %% CHECK AND INTERPRET
assert np.max(abs(amplitude - analytic)) < 3e-6
print("A single finite slit has a continuum of outgoing angular components.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(angles, abs(amplitude)**2, label='Numerical integral')
    ax.plot(angles, analytic**2, '--', label='sinc squared')
    ax.set(xlabel='Angle from normal (deg)', ylabel='Normalized far-field shape')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.1 - Resolve a single opening by direct far-field quadrature')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
