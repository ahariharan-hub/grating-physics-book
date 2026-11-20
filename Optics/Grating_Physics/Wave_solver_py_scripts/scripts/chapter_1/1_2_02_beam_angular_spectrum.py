# -*- coding: utf-8 -*-
"""1.2, worked example 2: Calculate the angular spectrum of a finite Gaussian beam.

Companion topic: A finite beam contains many directions.
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
waist = 3.0
window = 48.0
samples = 2048
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
fx = np.fft.fftshift(np.fft.fftfreq(samples, dx))

# %% NUMERICAL CALCULATION
field = np.exp(-(x / waist)**2)
spectrum = dx * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field)))
analytic = np.sqrt(np.pi) * waist * np.exp(-(np.pi * waist * fx)**2)

# %% CHECK AND INTERPRET
assert np.max(abs(spectrum - analytic)) < 1e-10
print("A smaller waist produces a wider spatial-frequency spectrum.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, abs(field)**2)
    axes[0].set(xlabel='x (um)', ylabel='Input intensity')
    axes[1].plot(fx, abs(spectrum), label='Sampled transform')
    axes[1].plot(fx, analytic, '--', label='Analytical Gaussian transform')
    axes[1].set(xlabel='fx (cycles/um)', ylabel='Spectral amplitude', xlim=(-0.5, 0.5))
    axes[1].legend()

    # %% NUMERICAL SUMMARY
    print('1.2 - Calculate the angular spectrum of a finite Gaussian beam')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
