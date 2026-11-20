# -*- coding: utf-8 -*-
"""7.4, worked example 3: Recover solved complex scattering amplitudes from sampled exit fields.

Companion topic: Physical scalar grating coefficients.
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
samples = 256
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
x = -period / 2 + np.arange(samples) * period / samples
field = solution.reconstruct(x, [-thickness])[0]
projected = np.mean(field[:, None] * np.exp(-1j * x[:, None] * solution.kx), axis=0)
difference = projected - solution.transmission

# %% CHECK AND INTERPRET
assert np.max(abs(difference)) < 1e-12
print("Projection recovers an existing scattering solution; it does not determine scattering independently.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(solution.orders, abs(projected), 'o')
    axes[0].set(xlabel='Order m', ylabel='Recovered magnitude')
    axes[1].plot(solution.orders, np.where(abs(projected) > 1e-6, np.angle(projected), np.nan), 'o')
    axes[1].set(xlabel='Order m', ylabel='Recovered phase (rad)')

    # %% NUMERICAL SUMMARY
    print('7.4 - Recover solved complex scattering amplitudes from sampled exit fields')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
