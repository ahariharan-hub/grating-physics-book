# -*- coding: utf-8 -*-
"""0.5, worked example 3: Compare a propagating-only cutoff with larger Fourier bases.

Companion topic: Can we solve using only propagating orders?.
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
from dataclasses import replace

# %% INPUT PARAMETERS
wavelength = 0.633
period = 1.5
opening_width = 0.75
thickness = 0.3
absorber_index = 1.5
extinction = 0.8
M = 8
cutoffs = [2, 4, 8, 16]
show_plot = True

# %% SETUP: coordinates, sampling and constants
base = AbsorbingGratingConfiguration(wavelength=wavelength, period=period,
    clear_width=opening_width, thickness=thickness, n_absorber=absorber_index, kappa=extinction)
coefficients = []
transmittances = []

# %% NUMERICAL CALCULATION
for cutoff in cutoffs:
    trial = solve_absorbing_grating(replace(base, M=cutoff))
    index = np.flatnonzero(trial.orders == 1)[0]
    coefficients.append(trial.transmission[index])
    transmittances.append(trial.transmittance)
coefficients = np.array(coefficients)
relative_difference = abs(coefficients - coefficients[-1]) / abs(coefficients[-1])

# %% CHECK AND INTERPRET
assert np.all(np.isfinite(relative_difference))
assert relative_difference[-1] == 0
print('Cutoff, relative C1 difference:', np.column_stack((cutoffs, relative_difference)))
print('The highest cutoff is a numerical reference, not exact truth.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].semilogy(cutoffs[:-1], np.maximum(relative_difference[:-1], 1e-16), 'o-')
    axes[0].set(xlabel='Fourier cutoff M', ylabel='Relative complex-coefficient difference')
    axes[1].plot(cutoffs, transmittances, 'o-')
    axes[1].set(xlabel='Fourier cutoff M', ylabel='Total transmittance')

    # %% NUMERICAL SUMMARY
    print('0.5 - Compare a propagating-only cutoff with larger Fourier bases')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
