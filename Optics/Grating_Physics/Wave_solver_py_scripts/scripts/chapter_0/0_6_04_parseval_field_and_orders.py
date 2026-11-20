# -*- coding: utf-8 -*-
"""0.6, worked example 4: Check Parseval and the importance of complex order phase.

Companion topic: One field, two representations.
Vacuum wavelength and lengths um. Normal-incidence scalar/TE model; air exteriors; outgoing -z; illustrative n+i*kappa.
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
from wave_optics.absorbing_grating import AbsorbingGratingConfiguration, solve_absorbing_grating

# %% INPUT PARAMETERS
wavelength = 0.633
period = 1.5
opening_width = 0.75
thickness = 0.3
absorber_index = 1.5
extinction = 0.8
M = 8
samples = 512
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
x = np.arange(samples) * period / samples
coefficients = solution.transmission
basis = np.exp(1j * solution.kx[:, None] * x)
field = coefficients @ basis
phase_removed = abs(coefficients) @ basis
mean_intensity = np.mean(abs(field)**2)
coefficient_norm = np.sum(abs(coefficients)**2)

# %% CHECK AND INTERPRET
assert np.allclose(mean_intensity, coefficient_norm)
assert np.linalg.norm(field - phase_removed) > 0.01
print('Spatial mean |U| squared:', mean_intensity, '; coefficient norm:', coefficient_norm)
print('This Parseval norm includes evanescent coefficients; it is not normal transmitted power.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(field)**2, label='Original complex coefficients')
    ax.plot(x, abs(phase_removed)**2, '--', label='Coefficient phase discarded')
    ax.set(xlabel='x over one period (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.6 - Check Parseval and the importance of complex order phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
