# -*- coding: utf-8 -*-
"""0.3, worked example 1: Check linear superposition with complex amplitudes.

Companion topic: Why can optical waves be added?.
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
second_amplitude = 0.6
relative_phase_degrees = 65.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
z = np.linspace(-3.0, 3.0, 601)
k = 2 * np.pi / wavelength
phase = np.deg2rad(relative_phase_degrees)

# %% NUMERICAL CALCULATION
first = np.exp(1j * k * z)
second = second_amplitude * np.exp(-1j * k * z + 1j * phase)
total = first + second
second_derivative = -k**2 * first - k**2 * second
residual = second_derivative + k**2 * total

# %% CHECK AND INTERPRET
assert np.max(abs(residual)) < 1e-12
print('Adding two homogeneous solutions preserves the field equation, not intensity additivity.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(z, abs(total)**2, label='Intensity of summed field')
    ax.plot(z, abs(first)**2 + abs(second)**2, '--', label='Sum of intensities')
    ax.set(xlabel='z (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.3 - Check linear superposition with complex amplitudes')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
