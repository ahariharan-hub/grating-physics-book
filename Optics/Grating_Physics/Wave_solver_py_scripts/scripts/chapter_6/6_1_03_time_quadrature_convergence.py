# -*- coding: utf-8 -*-
"""6.1, worked example 3: Check detector time-sampling convergence.

Companion topic: Numerically average one optical field.
Lengths are micrometres. Wavelength is in the medium. Time convention: exp(-i omega t).
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
from scipy.integrate import trapezoid

# %% INPUT PARAMETERS
amplitude = 1.0
phase = 0.4
exposure = 0.7
sample_counts = np.array([11, 21, 41, 81, 161])
show_plot = True

# %% SETUP: coordinates, sampling and constants
errors = []
analytic = amplitude**2 * (0.5 + (np.sin(4 * np.pi * exposure - 2 * phase) + np.sin(2 * phase)) / (8 * np.pi * exposure))

# %% NUMERICAL CALCULATION
for count in sample_counts:
    time = np.linspace(0, exposure, count)
    field = amplitude * np.cos(phase - 2 * np.pi * time)
    numerical = trapezoid(field**2, time) / exposure
    errors.append(abs(numerical - analytic))
errors = np.array(errors)

# %% CHECK AND INTERPRET
assert np.all(np.diff(errors) < 0)
print("Time quadrature must resolve the square of the oscillating field, not just its envelope.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.loglog(sample_counts, errors, 'o-')
    ax.set(xlabel='Time samples', ylabel='Absolute mean-square integration error')

    # %% NUMERICAL SUMMARY
    print('6.1 - Check detector time-sampling convergence')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
