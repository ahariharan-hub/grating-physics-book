# -*- coding: utf-8 -*-
"""8.1, worked example 3: Implement and finite-difference-check the Green normal derivative.

Companion topic: Boundary reconstruction needs normal derivatives.
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
wavelength = 1.0
observation = np.array([0.5, 0.0, 3.0])
finite_step = 1e-5
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-3, 3, 301)
boundary = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
normal = np.array([0.0, 0.0, 1.0])
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
difference = boundary - observation
radius = np.linalg.norm(difference, axis=1)
analytic = np.exp(1j * k * radius) * (1j * k * radius - 1) * (difference @ normal) / (4 * np.pi * radius**3)
plus_radius = np.linalg.norm(boundary + finite_step * normal - observation, axis=1)
minus_radius = np.linalg.norm(boundary - finite_step * normal - observation, axis=1)
numerical = (np.exp(1j * k * plus_radius) / plus_radius - np.exp(1j * k * minus_radius) / minus_radius) / (8 * np.pi * finite_step)

# %% CHECK AND INTERPRET
assert np.max(abs(numerical - analytic)) < 1e-8
print("The sign follows the stated boundary normal; reversing the normal reverses the derivative.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, analytic.real, label='Analytical real derivative')
    ax.plot(x, numerical.real, '--', label='Central-difference check')
    ax.set(xlabel='Boundary x (um)', ylabel='Normal derivative (inverse um squared)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.1 - Implement and finite-difference-check the Green normal derivative')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
