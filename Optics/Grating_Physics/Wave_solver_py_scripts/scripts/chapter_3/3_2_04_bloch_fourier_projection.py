# -*- coding: utf-8 -*-
"""3.2, worked example 4: Remove the incident carrier before projecting order labels.

Companion topic: Selected orders and phase continuity.
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
period = 4.0
incident_kx = 0.7
selected_order = -2
samples = 256
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.arange(samples) * period / samples
orders = np.arange(-5, 6)

# %% NUMERICAL CALCULATION
field = np.exp(1j * (incident_kx + selected_order * 2 * np.pi / period) * x)
periodic_part = field * np.exp(-1j * incident_kx * x)
coefficients = np.mean(periodic_part[:, None] * np.exp(-2j * np.pi * x[:, None] * orders / period), axis=0)

# %% CHECK AND INTERPRET
assert np.isclose(coefficients[orders == selected_order][0], 1)
assert np.max(abs(coefficients[orders != selected_order])) < 1e-12
print("Projection selects the integer Fourier label after carrier removal.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.stem(orders, abs(coefficients))
    ax.set(xlabel='Fourier order m', ylabel='Coefficient magnitude')

    # %% NUMERICAL SUMMARY
    print('3.2 - Remove the incident carrier before projecting order labels')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
