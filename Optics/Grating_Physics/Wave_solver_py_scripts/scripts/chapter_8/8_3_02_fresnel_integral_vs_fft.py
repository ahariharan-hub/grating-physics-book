# -*- coding: utf-8 -*-
"""8.3, worked example 2: Evaluate Fresnel propagation by a spatial integral and a Fourier multiplier.

Companion topic: Fresnel keeps quadratic path variation.
Two-transverse-dimensional paraxial propagation of a prescribed Gaussian boundary. Medium wavelength and lengths um.
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
waist = 2.5
distance = 8.0
window = 32.0
samples = 256
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
xx, yy = np.meshgrid(x, x)
k = 2 * np.pi / wavelength
frequencies = np.fft.fftfreq(samples, dx)
fx, fy = np.meshgrid(frequencies, frequencies)
indices = np.arange(samples // 2 - 32, samples // 2 + 33, 4)

# %% NUMERICAL CALCULATION
boundary = np.exp(-(xx**2 + yy**2) / waist**2)
multiplier = np.exp(1j * k * distance - 1j * np.pi * wavelength * distance * (fx**2 + fy**2))
fft_field = np.fft.ifft2(np.fft.fft2(boundary) * multiplier)
integral = []
for x_observer in x[indices]:
    kernel = np.exp(1j * k * distance) / (1j * wavelength * distance) * np.exp(1j * k * ((x_observer - xx)**2 + yy**2) / (2 * distance))
    integral.append(np.sum(boundary * kernel) * dx**2)
integral = np.array(integral)
fft_cut = fft_field[samples // 2, indices]
relative_difference = np.linalg.norm(integral - fft_cut) / np.linalg.norm(integral)

# %% CHECK AND INTERPRET
assert relative_difference < 1e-4
print("Spatial/Fourier Fresnel complex-field difference:", relative_difference)
print("Agreement verifies the two Fresnel implementations, not the physical accuracy of the approximation.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x[indices], integral.real, 'o', label='Direct Fresnel integral')
    ax.plot(x[indices], fft_cut.real, '--', label='Fresnel FFT multiplier')
    ax.set(xlabel='Detector x (um)', ylabel='Real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.3 - Evaluate Fresnel propagation by a spatial integral and a Fourier multiplier')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
