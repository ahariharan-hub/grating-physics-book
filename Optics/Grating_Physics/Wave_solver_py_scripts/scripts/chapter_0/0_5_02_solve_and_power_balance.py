# -*- coding: utf-8 -*-
"""0.5, worked example 2: Solve a physical grating and inspect its power budget.

Companion topic: What field does the solver calculate?.
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
reflected = np.sum(solution.reflection_efficiency)
transmitted = np.sum(solution.transmission_efficiency)
absorbed = solution.absorption
x = np.linspace(-period / 2, period / 2, 401)
exit_field = solution.reconstruct(x, [-thickness])[0]

# %% CHECK AND INTERPRET
assert abs(reflected + transmitted + absorbed - 1) < 1e-8
print('R, T, A:', reflected, transmitted, absorbed)
print('Boundary residual:', solution.boundary_residual)
print('Conservation and small matching residual do not certify Fourier-basis convergence.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, abs(exit_field)**2)
    axes[0].set(xlabel='x (um)', ylabel='Exit relative intensity')
    axes[1].bar(['R', 'T', 'A'], [reflected, transmitted, absorbed])
    axes[1].set(ylabel='Incident power fraction')

    # %% NUMERICAL SUMMARY
    print('0.5 - Solve a physical grating and inspect its power budget')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
