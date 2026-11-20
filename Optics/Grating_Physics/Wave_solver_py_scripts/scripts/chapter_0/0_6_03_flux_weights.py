# -*- coding: utf-8 -*-
"""0.6, worked example 3: Distinguish coefficient magnitude squared from normal power.

Companion topic: Where does the incident power go?.
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
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
coefficients = solution.transmission
incident_q = 2 * np.pi / wavelength
flux_factor = solution.beta_output.real / incident_q
efficiency = abs(coefficients)**2 * flux_factor

# %% CHECK AND INTERPRET
assert np.allclose(efficiency, solution.transmission_efficiency)
assert np.all(efficiency[solution.output_kinds == 'evanescent'] == 0)
print('Sum of transmitted normal efficiencies:', efficiency.sum())

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(solution.orders, abs(coefficients)**2, 'o-', label='Magnitude squared')
    ax.plot(solution.orders, efficiency, 's-', label='Normal power efficiency')
    ax.set(xlabel='Order m', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.6 - Distinguish coefficient magnitude squared from normal power')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
