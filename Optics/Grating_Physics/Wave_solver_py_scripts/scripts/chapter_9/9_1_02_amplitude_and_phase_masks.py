# -*- coding: utf-8 -*-
"""9.1, worked example 2: Compare two masks with different complex transmission but simple geometry.

Companion topic: What defines the complex exit field?.
Prescribed unit-cell transmissions, normal incidence, lengths um. Fourier magnitudes are not automatically power efficiencies.
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

# %% INPUT PARAMETERS
period = 4.0
phase_step_degrees = 180.0
samples = 1024
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = -period / 2 + (np.arange(samples) + 0.5) * period / samples
orders = np.arange(-6, 7)
first_region = x < 0

# %% NUMERICAL CALCULATION
amplitude_mask = first_region.astype(complex)
phase_mask = np.where(first_region, 1.0, np.exp(1j * np.deg2rad(phase_step_degrees)))
basis = np.exp(-2j * np.pi * x[:, None] * orders / period)
amplitude_coefficients = np.mean(amplitude_mask[:, None] * basis, axis=0)
phase_coefficients = np.mean(phase_mask[:, None] * basis, axis=0)

# %% CHECK AND INTERPRET
assert np.allclose(abs(phase_mask), 1)
assert np.isclose(phase_coefficients[orders == 0][0], (1 + np.exp(1j * np.deg2rad(phase_step_degrees))) / 2)
print("The phase mask transmits unit local intensity; its zero-order field cancels only at a pi step.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(orders, abs(amplitude_coefficients), 'o-', label='Amplitude mask')
    ax.plot(orders, abs(phase_coefficients), 's-', label='Pi-phase mask')
    ax.set(xlabel='Order m', ylabel='Fourier coefficient magnitude')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.1 - Compare two masks with different complex transmission but simple geometry')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
