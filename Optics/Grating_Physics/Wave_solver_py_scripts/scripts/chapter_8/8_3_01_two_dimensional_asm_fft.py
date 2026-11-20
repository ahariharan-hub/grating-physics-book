# -*- coding: utf-8 -*-
"""8.3, worked example 1: Implement two-dimensional ASM using FFT frequencies and the outgoing branch.

Companion topic: Full propagation and the ASM multiplier.
Prescribed square aperture; medium wavelength and lengths um; outgoing distance s=-z. FFT domain is periodic.
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
aperture_width = 4.0
distance = 10.0
window = 32.0
samples = 256
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
xx, yy = np.meshgrid(x, x)
frequencies = np.fft.fftfreq(samples, dx)
fx, fy = np.meshgrid(frequencies, frequencies)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
boundary = ((abs(xx) < aperture_width / 2) & (abs(yy) < aperture_width / 2)).astype(complex)
spectrum = np.fft.fft2(boundary)
q = k * np.sqrt((1 - wavelength**2 * (fx**2 + fy**2)).astype(complex))
multiplier = np.exp(1j * q * distance)
output = np.fft.ifft2(spectrum * multiplier)
roundtrip = np.fft.ifft2(spectrum)

# %% CHECK AND INTERPRET
assert np.allclose(roundtrip, boundary)
assert np.all(abs(multiplier) <= 1 + 1e-12)
print("Grid step (um):", dx, "; evanescent components decay instead of growing.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, values, title in zip(axes, [abs(boundary)**2, abs(output)**2], ['Input intensity', 'ASM output intensity']):
        image = ax.imshow(values, origin='lower', extent=[x[0], x[-1], x[0], x[-1]])
        ax.set(xlabel='x (um)', ylabel='y (um)', title=title)
        figure.colorbar(image, ax=ax)

    # %% NUMERICAL SUMMARY
    print('8.3 - Implement two-dimensional ASM using FFT frequencies and the outgoing branch')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
