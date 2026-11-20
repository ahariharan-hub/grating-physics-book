# -*- coding: utf-8 -*-
"""3.1, worked example 3: Estimate incident direction by unwrapping sampled phase.

Companion topic: Explore the surface phase change.
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
incidence_degrees = 35.0
samples = 201
window = 8.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-window / 2, window / 2, samples)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
true_kx = k * np.sin(np.deg2rad(incidence_degrees))
field = np.exp(1j * (true_kx * x + 0.8))
unwrapped = np.unwrap(np.angle(field))
fitted_slope, intercept = np.polyfit(x, unwrapped, 1)
estimated_angle = np.rad2deg(np.arcsin(fitted_slope / k))

# %% CHECK AND INTERPRET
assert np.isclose(estimated_angle, incidence_degrees)
print("Recovered incidence (deg):", estimated_angle)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, np.angle(field), label='Wrapped phase')
    ax.plot(x, unwrapped, label='Unwrapped phase')
    ax.plot(x, fitted_slope * x + intercept, '--', label='Linear fit')
    ax.set(xlabel='x (um)', ylabel='Phase (rad)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('3.1 - Estimate incident direction by unwrapping sampled phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
