# -*- coding: utf-8 -*-
"""3.2, worked example 2: Measure phase mismatch for a noninteger trial order.

Companion topic: Test the seam in phase cycles.
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
trial_orders = np.linspace(-2, 2, 801)
show_plot = True

# %% SETUP: coordinates, sampling and constants
integer_orders = np.arange(-2, 3)

# %% NUMERICAL CALCULATION
seam_ratio = np.exp(2j * np.pi * trial_orders)
seam_error = abs(seam_ratio - 1)
integer_error = abs(np.exp(2j * np.pi * integer_orders) - 1)

# %% CHECK AND INTERPRET
assert np.max(integer_error) < 1e-12
print("Relative seam error vanishes at every integer, including negative integers.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(trial_orders, seam_error)
    ax.plot(integer_orders, integer_error, 'o')
    ax.set(xlabel='Trial order label', ylabel='Absolute complex seam mismatch')

    # %% NUMERICAL SUMMARY
    print('3.2 - Measure phase mismatch for a noninteger trial order')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
