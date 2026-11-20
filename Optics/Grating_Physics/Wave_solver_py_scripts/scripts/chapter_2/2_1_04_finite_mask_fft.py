# -*- coding: utf-8 -*-
"""2.1, worked example 4: Compute a finite grating spectrum from sampled real-space openings.

Companion topic: Numerical representation of finite openings.
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
width = 2.0
openings = 5
window = 64.0
samples = 4096
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
positions = (np.arange(openings) - (openings - 1) / 2) * period
fx = np.fft.fftshift(np.fft.fftfreq(samples, dx))

# %% NUMERICAL CALCULATION
# Midpoint samples avoid double-counting an opening edge.
x = x + dx / 2
mask = np.any(abs(x[:, None] - positions) < width / 2, axis=1).astype(float)
spectrum = dx * np.fft.fftshift(np.fft.fft(mask))
intensity = abs(spectrum)**2 / (openings * width)**2

# %% CHECK AND INTERPRET
assert np.isclose(mask.sum() * dx, openings * width)
assert np.isclose(intensity[samples // 2], 1)
print("FFT spectrum retains absolute aperture-area scaling before normalization.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, mask)
    axes[0].set(xlabel='x (um)', ylabel='Amplitude transmission')
    axes[1].plot(fx, intensity)
    axes[1].set(xlabel='fx (cycles/um)', ylabel='Normalized spectral intensity', xlim=(-0.8, 0.8))

    # %% NUMERICAL SUMMARY
    print('2.1 - Compute a finite grating spectrum from sampled real-space openings')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
