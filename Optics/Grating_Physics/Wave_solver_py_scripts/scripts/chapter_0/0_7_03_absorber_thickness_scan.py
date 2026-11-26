# -*- coding: utf-8 -*-
"""0.7, worked example 3: Test whether greater absorber thickness approaches an ideal mask.

Companion topic: Does thicker mean closer to ideal?.
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
thicknesses = np.linspace(0.1, 0.7, 7)
selected_order = 1
show_plot = True

# %% SETUP: coordinates, sampling and constants
base = AbsorbingGratingConfiguration(wavelength=wavelength, period=period,
    clear_width=opening_width, n_absorber=absorber_index, kappa=extinction, M=M)
ideal_coefficient = opening_width / period * np.sinc(selected_order * opening_width / period)
physical_coefficients = []
ideal_coefficients = []

# %% NUMERICAL CALCULATION
for height in thicknesses:
    trial = solve_absorbing_grating(replace(base, thickness=float(height)))
    coefficient = trial.transmission[trial.orders == selected_order][0]
    beta_m = trial.beta_output[trial.orders == selected_order][0]
    physical_coefficients.append(coefficient)
    ideal_coefficients.append(ideal_coefficient * np.exp(1j * beta_m * height))
physical_coefficients = np.array(physical_coefficients)
ideal_coefficients = np.array(ideal_coefficients)
differences = abs(physical_coefficients - ideal_coefficients)

# %% CHECK AND INTERPRET
assert np.all(np.isfinite(differences))
print('Best sampled thickness (um):', thicknesses[np.argmin(differences)])
print('A thickness scan is not a proof that opacity alone produces an ideal mask.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(thicknesses, differences, 'o-')
    ax.set(xlabel='Absorber thickness (um)', ylabel='Complex difference from ideal selected coefficient')

    # %% NUMERICAL SUMMARY
    print('0.7 - Test whether greater absorber thickness approaches an ideal mask')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
