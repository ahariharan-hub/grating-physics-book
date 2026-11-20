# -*- coding: utf-8 -*-
"""0.6, worked example 2: Advance complex coefficients to a new reference plane.

Companion topic: What changes when we move the output reference plane?.
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
depths = np.linspace(0.0, 1.0, 101)
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = AbsorbingGratingConfiguration(
    wavelength=wavelength, period=period, clear_width=opening_width,
    thickness=thickness, n_absorber=absorber_index, kappa=extinction, M=M)
solution = solve_absorbing_grating(configuration)

# %% NUMERICAL CALCULATION
exit_coefficients = solution.transmission
q = solution.beta_output
advanced = exit_coefficients[None, :] * np.exp(1j * depths[:, None] * q)
direct = solution.at_reference_plane(-thickness - depths[-1])
propagating = solution.output_kinds == 'propagating'

# %% CHECK AND INTERPRET
assert np.allclose(advanced[-1], direct)
assert np.allclose(abs(advanced[:, propagating]), abs(exit_coefficients[propagating]))
print('Reference-plane translation rotates propagating coefficients and attenuates evanescent ones.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for order in [0, 1, 3]:
        index = np.flatnonzero(solution.orders == order)[0]
        ax.plot(depths, abs(advanced[:, index]), label=f'Order {order}')
    ax.set(xlabel='Depth below exit (um)', ylabel='Coefficient magnitude')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.6 - Advance complex coefficients to a new reference plane')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
