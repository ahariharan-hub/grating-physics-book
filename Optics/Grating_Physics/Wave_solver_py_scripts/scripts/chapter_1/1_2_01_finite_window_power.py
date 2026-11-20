# -*- coding: utf-8 -*-
"""1.2, worked example 1: Integrate uniform and Gaussian illumination over finite windows.

Companion topic: Width and integrated intensity.
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
waist = 3.0
half_windows = np.linspace(1, 15, 40)
show_plot = True

# %% SETUP: coordinates, sampling and constants
uniform_integrals = []
gaussian_integrals = []

# %% NUMERICAL CALCULATION
for half_window in half_windows:
    x = np.linspace(-half_window, half_window, 2001)
    uniform_integrals.append(trapezoid(np.ones_like(x), x))
    gaussian_integrals.append(trapezoid(np.exp(-2 * (x / waist)**2), x))
limit = waist * np.sqrt(np.pi / 2)

# %% CHECK AND INTERPRET
assert np.isclose(gaussian_integrals[-1], limit, rtol=1e-6)
assert np.isclose(uniform_integrals[-1], 2 * half_windows[-1])
print("Finite Gaussian line-integral limit:", limit)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(half_windows, uniform_integrals, label='Uniform wave')
    ax.plot(half_windows, gaussian_integrals, label='Gaussian envelope')
    ax.axhline(limit, color='gray', linestyle=':')
    ax.set(xlabel='Half-window (um)', ylabel='Integrated relative intensity (um)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('1.2 - Integrate uniform and Gaussian illumination over finite windows')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
