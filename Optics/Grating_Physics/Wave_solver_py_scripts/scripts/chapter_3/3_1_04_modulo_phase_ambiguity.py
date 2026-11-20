# -*- coding: utf-8 -*-
"""3.1, worked example 4: Show why one complex translation factor cannot count phase cycles.

Companion topic: Phase cycles and the order rule.
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
base_kx = 0.3
integers = np.array([-1, 0, 1, 2])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(0, period, 501)

# %% NUMERICAL CALCULATION
candidate_kx = base_kx + integers * 2 * np.pi / period
phases = candidate_kx[:, None] * x
translation_factors = np.exp(1j * candidate_kx * period)

# %% CHECK AND INTERPRET
assert np.allclose(translation_factors, translation_factors[0])
print("Different unwrapped cycles give the same modulo-2pi translation phasor.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for order, phase in zip(integers, phases):
        ax.plot(x, phase / (2 * np.pi), label=f'Extra cycles {order}')
    ax.set(xlabel='Surface displacement (um)', ylabel='Unwrapped phase cycles')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('3.1 - Show why one complex translation factor cannot count phase cycles')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
