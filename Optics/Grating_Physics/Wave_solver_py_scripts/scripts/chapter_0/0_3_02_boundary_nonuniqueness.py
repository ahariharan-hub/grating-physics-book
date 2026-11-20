# -*- coding: utf-8 -*-
"""0.3, worked example 2: Compare fields that share the same boundary value.

Companion topic: Does one boundary value select one field?.
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
backward_fractions = np.array([0.0, 0.3, 0.5])
show_plot = True

# %% SETUP: coordinates, sampling and constants
z = np.linspace(0.0, 5.0, 601)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
fields = ((1 - backward_fractions[:, None]) * np.exp(1j * k * z)
          + backward_fractions[:, None] * np.exp(-1j * k * z))
boundary_derivatives = 1j * k * (1 - 2 * backward_fractions)

# %% CHECK AND INTERPRET
assert np.allclose(fields[:, 0], 1.0)
assert not np.allclose(boundary_derivatives, boundary_derivatives[0])
print('Boundary derivatives:', boundary_derivatives)
print('Radiation conditions and appropriate boundary data distinguish these solutions.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for fraction, field in zip(backward_fractions, fields):
        ax.plot(z, field.real, label=f'Backward fraction {fraction:g}')
    ax.set(xlabel='z (um)', ylabel='Real field; all U(0)=1')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.3 - Compare fields that share the same boundary value')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
