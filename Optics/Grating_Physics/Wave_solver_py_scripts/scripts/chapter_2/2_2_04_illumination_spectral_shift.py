# -*- coding: utf-8 -*-
"""2.2, worked example 4: Measure the spectral shift caused by an incident phase ramp.

Companion topic: Illumination and the angular spectrum.
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
window = 32.0
samples = 1024
waist = 3.0
shift_bins = 5
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = np.arange(samples) * dx - window / 2
fx = np.fft.fftfreq(samples, dx)
incident_frequency = shift_bins / window

# %% NUMERICAL CALCULATION
boundary = np.exp(-(x / waist)**2)
tilted = boundary * np.exp(2j * np.pi * incident_frequency * x)
original_spectrum = abs(np.fft.fft(boundary))
shifted_spectrum = abs(np.fft.fft(tilted))

# %% CHECK AND INTERPRET
assert np.allclose(shifted_spectrum, np.roll(original_spectrum, shift_bins), atol=1e-10)
print("Measured spectral shift (cycles/um):", fx[np.argmax(shifted_spectrum)])

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(np.fft.fftshift(fx), np.fft.fftshift(original_spectrum), label='Untilted boundary')
    ax.plot(np.fft.fftshift(fx), np.fft.fftshift(shifted_spectrum), label='Tilted boundary')
    ax.set(xlabel='fx (cycles/um)', ylabel='Spectral magnitude', xlim=(-0.4, 0.5))
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.2 - Measure the spectral shift caused by an incident phase ramp')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
