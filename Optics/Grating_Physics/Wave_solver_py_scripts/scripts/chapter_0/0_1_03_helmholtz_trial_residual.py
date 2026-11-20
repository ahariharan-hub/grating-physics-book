# -*- coding: utf-8 -*-
"""0.1, worked example 3: Test an allowed and an incorrect spatial wavenumber.

Companion topic: Does a trial field satisfy Helmholtz?.
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
trial_ratios = np.array([-1.0, 1.0, 1.25])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-2.0, 2.0, 501)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
fields = np.exp(1j * trial_ratios[:, None] * k * x)
second_derivatives = -(trial_ratios[:, None] * k)**2 * fields
residuals = second_derivatives + k**2 * fields
normalized_residual = np.max(abs(residuals), axis=1) / k**2

# %% CHECK AND INTERPRET
assert np.allclose(normalized_residual[:2], 0.0)
assert normalized_residual[2] > 0.1
print('q/k, normalized residual:', np.column_stack((trial_ratios, normalized_residual)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for ratio, residual in zip(trial_ratios, residuals):
        ax.plot(x, residual.real / k**2, label=f'q/k={ratio:g}')
    ax.set(xlabel='x (um)', ylabel='Re residual / k squared')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.1 - Test an allowed and an incorrect spatial wavenumber')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
