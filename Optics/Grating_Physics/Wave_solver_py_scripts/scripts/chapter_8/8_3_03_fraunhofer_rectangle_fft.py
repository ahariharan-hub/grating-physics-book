# -*- coding: utf-8 -*-
"""8.3, worked example 3: Recover rectangular-aperture Fraunhofer diffraction with physical Fourier scaling.

Companion topic: Fraunhofer omits aperture-coordinate curvature.
Prescribed rectangular aperture, medium wavelength and lengths um. Far-field and paraxial assumptions must be checked separately.
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
width_x = 10.0
width_y = 6.0
distance = 1000.0
window = 64.0
samples = 512
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x0 = (-samples / 2 + 0.5) * dx
x = x0 + np.arange(samples) * dx
xx, yy = np.meshgrid(x, x)
frequencies = np.fft.fftshift(np.fft.fftfreq(samples, dx))
fx, fy = np.meshgrid(frequencies, frequencies)

# %% NUMERICAL CALCULATION
boundary = ((abs(xx) < width_x / 2) & (abs(yy) < width_y / 2)).astype(float)
transform = dx**2 * np.fft.fftshift(np.fft.fft2(boundary)) * np.exp(-2j * np.pi * (fx * x0 + fy * x0))
analytic = width_x * width_y * np.sinc(width_x * fx) * np.sinc(width_y * fy)
detector_coordinate = wavelength * distance * frequencies
row = samples // 2
selected = abs(frequencies) <= 0.2
normalized_error = np.max(abs(transform[row, selected] - analytic[row, selected])) / (width_x * width_y)
field_cut = np.exp(2j * np.pi * distance / wavelength) / (1j * wavelength * distance) * np.exp(1j * np.pi * detector_coordinate**2 / (wavelength * distance)) * transform[row]

# %% CHECK AND INTERPRET
assert np.isclose(boundary.sum() * dx**2, width_x * width_y)
assert normalized_error < 0.002
print("Main-region Fourier-integral error / aperture area:", normalized_error)
print("Physical coordinate X=lambda*s*fx uses the paraxial Fraunhofer mapping.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(detector_coordinate[selected], abs(field_cut[selected])**2, label='Sampled aperture FFT')
    ax.plot(detector_coordinate[selected], abs(analytic[row, selected] / (wavelength * distance))**2, '--', label='Analytical rectangular aperture')
    ax.set(xlabel='Detector X (um)', ylabel='Relative scalar intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.3 - Recover rectangular-aperture Fraunhofer diffraction with physical Fourier scaling')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
