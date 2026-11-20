# -*- coding: utf-8 -*-
"""0.5, worked example 4: Rebuild the solved field with an explicit Fourier sum.

Companion topic: How does the same problem look in Fourier space?.
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
output_depth = 0.2
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
x = np.linspace(-period / 2, period / 2, 601)
z_reference = -thickness - output_depth
coefficients = solution.at_reference_plane(z_reference)
field = np.zeros(x.size, dtype=complex)
for kx, coefficient in zip(solution.kx, coefficients):
    field += coefficient * np.exp(1j * kx * x)
backend_field = solution.reconstruct(x, [z_reference])[0]

# %% CHECK AND INTERPRET
error = np.max(abs(field - backend_field))
assert error < 1e-12
print('Explicit sum / backend reconstruction mismatch:', error)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, field.real, label='Explicit sum: real')
    ax.plot(x, field.imag, label='Explicit sum: imaginary')
    ax.plot(x, backend_field.real, '--', label='Backend real-field check')
    ax.set(xlabel='x (um)', ylabel='Complex field components')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.5 - Rebuild the solved field with an explicit Fourier sum')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
