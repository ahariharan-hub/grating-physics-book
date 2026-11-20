# -*- coding: utf-8 -*-
"""9.1, worked example 4: Extract a physical exit field and move its reference plane.

Companion topic: A physical model supplies the exit field.
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
output_depth = 0.25
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
x = np.linspace(-period / 2, period / 2, 601)
exit_field = solution.reconstruct(x, [-thickness])[0]
exit_coefficients = solution.transmission
propagated_coefficients = exit_coefficients * np.exp(1j * solution.beta_output * output_depth)
explicit_output = propagated_coefficients @ np.exp(1j * solution.kx[:, None] * x)
backend_output = solution.reconstruct(x, [-thickness - output_depth])[0]

# %% CHECK AND INTERPRET
assert np.allclose(explicit_output, backend_output)
print("The exit field and a later reference-plane field must not be given the same label without propagation.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(exit_field)**2, label='At physical grating exit')
    ax.plot(x, abs(explicit_output)**2, label='At downstream reference plane')
    ax.set(xlabel='x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.1 - Extract a physical exit field and move its reference plane')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
