# -*- coding: utf-8 -*-
"""5.2, worked example 4: Verify Harvey and Ewald sphere equations for the same 3D orders.

Companion topic: Equivalent diffraction-coordinate pictures.
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
period = 4.0
incident = np.array([0.3, 0.4, -np.sqrt(0.75)])
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-4, 5)
alpha = incident[0] + orders * wavelength / period
radicand = 1 - alpha**2 - incident[1]**2

# %% NUMERICAL CALCULATION
valid = radicand > 0
outgoing = np.column_stack([alpha[valid], np.full(valid.sum(), incident[1]), -np.sqrt(radicand[valid])])
change = outgoing - incident
direction_residual = np.sum(outgoing**2, axis=1) - 1
translated_residual = np.sum((change + incident)**2, axis=1) - 1

# %% CHECK AND INTERPRET
assert np.max(abs(direction_residual)) < 1e-12
assert np.max(abs(translated_residual)) < 1e-12
print("Both sphere equations describe the same allowed directions.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders[valid], direction_residual, 'o-', label='Direction-sphere residual')
    ax.plot(orders[valid], translated_residual, 'x--', label='Translated Ewald residual')
    ax.set(xlabel='Order m', ylabel='Dimensionless sphere residual')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.2 - Verify Harvey and Ewald sphere equations for the same 3D orders')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
