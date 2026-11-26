# -*- coding: utf-8 -*-
"""0.7, worked example 2: Compare both fields at the exit after prescribing the thin mask at z=0.

Companion topic: Read the field and coefficient views.
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
physical = solution.transmission
t_m = opening_width / period * np.sinc(solution.orders * opening_width / period)
ideal = t_m * np.exp(1j * solution.beta_output * thickness)
x = np.linspace(-period / 2, period / 2, 601)
physical_field = physical @ np.exp(1j * solution.kx[:, None] * x)
ideal_truncated = ideal @ np.exp(1j * solution.kx[:, None] * x)

# %% CHECK AND INTERPRET
assert np.allclose(physical_field, solution.reconstruct(x, [-thickness])[0])
print('The thin mask starts at z=0; each order propagates to the common exit plane z=-h.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(x, physical_field.real, label='Physical real field')
    axes[0].plot(x, ideal_truncated.real, '--', label='Ideal mask propagated from z=0')
    axes[0].set(xlabel='x (um)', ylabel='Real field at the physical exit')
    axes[0].legend()
    axes[1].plot(solution.orders, abs(physical - ideal), 'o-')
    axes[1].set(xlabel='Order m', ylabel='Complex coefficient difference magnitude')

    # %% NUMERICAL SUMMARY
    print('0.7 - Compare both fields at the exit after prescribing the thin mask at z=0')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
