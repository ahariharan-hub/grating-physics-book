# -*- coding: utf-8 -*-
"""0.1, worked example 4: Check the convergence of a numerical second derivative.

Companion topic: Numerically verify the Helmholtz equation.
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

# %% INPUT PARAMETERS
wavelength = 2.0
sample_counts = np.array([41, 81, 161, 321])
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
errors = []
steps = []

# %% NUMERICAL CALCULATION
for count in sample_counts:
    x = np.linspace(0.0, 4.0, count)
    dx = x[1] - x[0]
    field = np.exp(1j * k * x)
    second = (field[2:] - 2 * field[1:-1] + field[:-2]) / dx**2
    residual = second + k**2 * field[1:-1]
    errors.append(np.max(abs(residual)) / k**2)
    steps.append(dx)
errors = np.array(errors)
steps = np.array(steps)

# %% CHECK AND INTERPRET
assert np.all(np.diff(errors) < 0)
assert 3.5 < errors[-2] / errors[-1] < 4.5
print('Halving dx reduces the central-difference error by approximately four.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.loglog(steps, errors, 'o-', label='Measured residual')
    ax.loglog(steps, errors[-1] * (steps / steps[-1])**2, '--', label='dx squared')
    ax.set(xlabel='Grid spacing (um)', ylabel='Normalized residual')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.1 - Check the convergence of a numerical second derivative')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
