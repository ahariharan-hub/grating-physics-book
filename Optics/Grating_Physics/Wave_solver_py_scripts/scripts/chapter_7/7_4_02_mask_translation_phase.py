# -*- coding: utf-8 -*-
"""7.4, worked example 2: Derive complex mask coefficients and their translation phase.

Companion topic: Where do complex coefficients come from?.
Dimensionless unit-cell coordinate; prescribed real binary transmission. No physical material solve.
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
clear_fraction = 0.4
shift_fraction = 0.15
samples = 20000
orders = np.arange(-5, 6)
show_plot = True

# %% SETUP: coordinates, sampling and constants
coordinate = -0.5 + (np.arange(samples) + 0.5) / samples

# %% NUMERICAL CALCULATION
centered = (abs(coordinate) < clear_fraction / 2).astype(float)
shifted = (abs((coordinate - shift_fraction + 0.5) % 1 - 0.5) < clear_fraction / 2).astype(float)
basis = np.exp(-2j * np.pi * coordinate[:, None] * orders)
center_coefficients = np.mean(centered[:, None] * basis, axis=0)
shifted_coefficients = np.mean(shifted[:, None] * basis, axis=0)
prediction = center_coefficients * np.exp(-2j * np.pi * orders * shift_fraction)

# %% CHECK AND INTERPRET
assert np.allclose(shifted_coefficients, prediction, atol=2 / samples)
print("Translation changes coefficient phase without changing its magnitude, within sampling error.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(orders, abs(center_coefficients), 'o-', label='Centered')
    axes[0].plot(orders, abs(shifted_coefficients), 'x--', label='Shifted')
    axes[0].set(xlabel='Order m', ylabel='Magnitude')
    axes[0].legend()
    axes[1].plot(orders, np.where(abs(shifted_coefficients) > 1e-6, np.angle(shifted_coefficients), np.nan), 'o')
    axes[1].set(xlabel='Order m', ylabel='Shifted coefficient phase (rad)')

    # %% NUMERICAL SUMMARY
    print('7.4 - Derive complex mask coefficients and their translation phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
