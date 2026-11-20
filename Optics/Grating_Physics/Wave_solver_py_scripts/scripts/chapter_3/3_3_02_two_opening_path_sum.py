# -*- coding: utf-8 -*-
"""3.3, worked example 2: Add two path contributions as complex amplitudes.

Companion topic: Path difference and constructive interference.
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
path_differences = np.linspace(-2, 2, 801)
show_plot = True

# %% SETUP: coordinates, sampling and constants
phase_difference = 2 * np.pi * path_differences / wavelength

# %% NUMERICAL CALCULATION
first = np.ones(path_differences.size, dtype=complex)
second = np.exp(1j * phase_difference)
intensity = abs(first + second)**2
expanded = 2 + 2 * np.cos(phase_difference)

# %% CHECK AND INTERPRET
assert np.allclose(intensity, expanded)
print("Integer-wavelength path differences are constructive; half-integer differences are destructive.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(path_differences / wavelength, intensity)
    ax.set(xlabel='Path difference / wavelength', ylabel='Relative intensity')

    # %% NUMERICAL SUMMARY
    print('3.3 - Add two path contributions as complex amplitudes')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
