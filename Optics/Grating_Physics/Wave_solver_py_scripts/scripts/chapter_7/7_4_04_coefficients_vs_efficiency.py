# -*- coding: utf-8 -*-
"""7.4, worked example 4: Apply the longitudinal flux factor for unequal exterior indices.

Companion topic: Complex coefficients versus diffraction efficiency.
Vacuum wavelength and lengths um; normal-incidence lossless scalar/TE grating; outgoing -z.
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
from wave_optics.scalar_grating import ScalarGratingConfiguration, solve_scalar_grating

# %% INPUT PARAMETERS
wavelength = 0.633
period = 1.5
thickness = 0.3
n_incident = 1.0
n_output = 1.4
M = 8
show_plot = True

# %% SETUP: coordinates, sampling and constants
configuration = ScalarGratingConfiguration(wavelength=wavelength, period=period, thickness=thickness,
    n_incident=n_incident, n_output=n_output, n_ridge=1.8, n_groove=1.0, M=M)
solution = solve_scalar_grating(configuration)

# %% NUMERICAL CALCULATION
incident_normal_component = 2 * np.pi / wavelength * n_incident
flux_factors = solution.beta_output.real / incident_normal_component
efficiencies = abs(solution.transmission)**2 * flux_factors

# %% CHECK AND INTERPRET
assert np.allclose(efficiencies, solution.transmission_efficiency)
assert abs(solution.power_balance - 1) < 1e-10
print("A coefficient modulus is not automatically a power fraction when normal flux factors differ.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(solution.orders, abs(solution.transmission)**2, 'o-', label='Magnitude squared')
    ax.plot(solution.orders, efficiencies, 's-', label='Normal transmitted efficiency')
    ax.set(xlabel='Order m', ylabel='Relative quantity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.4 - Apply the longitudinal flux factor for unequal exterior indices')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
