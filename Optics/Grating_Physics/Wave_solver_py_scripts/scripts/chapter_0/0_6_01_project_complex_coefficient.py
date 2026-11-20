# -*- coding: utf-8 -*-
"""0.6, worked example 1: Recover one coefficient by numerical projection.

Companion topic: Can samples recover one complex coefficient?.
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
selected_order = 1
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
kx_selected = 2 * np.pi * selected_order / period
projected = np.mean(field * np.exp(-1j * kx_selected * x))
direct = solution.transmission[solution.orders == selected_order][0]

# %% CHECK AND INTERPRET
assert abs(projected - direct) < 1e-12
print('Projected coefficient:', projected, '; solved coefficient:', direct)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    integrand = field * np.exp(-1j * kx_selected * x)
    ax.plot(x, integrand.real, label='Real projection integrand')
    ax.plot(x, integrand.imag, label='Imaginary projection integrand')
    ax.set(xlabel='x over one period (um)', ylabel='Complex projection integrand')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.6 - Recover one coefficient by numerical projection')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
