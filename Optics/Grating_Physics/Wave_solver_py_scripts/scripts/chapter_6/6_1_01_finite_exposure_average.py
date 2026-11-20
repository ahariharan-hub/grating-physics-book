# -*- coding: utf-8 -*-
"""6.1, worked example 1: Integrate the real-field square over a finite detector exposure.

Companion topic: Time averaging and relative intensity.
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
amplitude = 1.0
phase_degrees = 40.0
exposures = np.linspace(0.05, 4, 161)
show_plot = True

# %% SETUP: coordinates, sampling and constants
phase = np.deg2rad(phase_degrees)
means = []

# %% NUMERICAL CALCULATION
for exposure in exposures:
    time = np.linspace(0, exposure, 2001)  # time in optical periods
    real_field = amplitude * np.cos(phase - 2 * np.pi * time)
    means.append(trapezoid(real_field**2, time) / exposure)
means = np.array(means)
analytic = amplitude**2 * (0.5 + (np.sin(4 * np.pi * exposures - 2 * phase) + np.sin(2 * phase)) / (8 * np.pi * exposures))

# %% CHECK AND INTERPRET
assert np.max(abs(means - analytic)) < 2e-5
print("Short exposure depends on initial phase; complete-cycle averages are A squared / 2.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(exposures, means, label='Numerical time integration')
    ax.axhline(amplitude**2 / 2, color='gray', linestyle='--', label='Long-exposure limit')
    ax.set(xlabel='Exposure / T', ylabel='Mean square of real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('6.1 - Integrate the real-field square over a finite detector exposure')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
