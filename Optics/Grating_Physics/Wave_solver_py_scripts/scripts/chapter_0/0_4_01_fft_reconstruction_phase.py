# -*- coding: utf-8 -*-
"""0.4, worked example 1: Reconstruct a complex field with and without spectral phase.

Companion topic: A complicated field can be built from plane waves.
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
samples = 512
window = 32.0
width = 3.0
shift = 4.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = (np.arange(samples) - samples // 2) * window / samples

# %% NUMERICAL CALCULATION
field = np.exp(-((x - shift) / width)**2) * np.exp(0.3j * x)
spectrum = np.fft.fft(field)
reconstructed = np.fft.ifft(spectrum)
phase_removed = np.fft.ifft(abs(spectrum))

# %% CHECK AND INTERPRET
assert np.max(abs(reconstructed - field)) < 1e-12
assert np.linalg.norm(phase_removed - field) > 0.1
print('Magnitude alone loses position and complex phase information.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(field), label='Original amplitude')
    ax.plot(x, abs(reconstructed), '--', label='Full complex spectrum')
    ax.plot(x, abs(phase_removed), label='Spectral phase removed')
    ax.set(xlabel='x (um)', ylabel='Amplitude')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.4 - Reconstruct a complex field with and without spectral phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
