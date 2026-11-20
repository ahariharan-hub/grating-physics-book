# -*- coding: utf-8 -*-
"""1.3, worked example 4: Find a plane-wave direction using a two-dimensional FFT.

Companion topic: Plane-wave direction in Fourier space.
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
samples = 128
window = 16.0
mode_x = 3
mode_z = 4
show_plot = True

# %% SETUP: coordinates, sampling and constants
coordinate = np.arange(samples) * window / samples
xx, zz = np.meshgrid(coordinate, coordinate)
frequency = np.fft.fftfreq(samples, window / samples)

# %% NUMERICAL CALCULATION
kx, kz = 2 * np.pi / window * np.array([mode_x, mode_z])
field = np.exp(1j * (kx * xx + kz * zz))
spectrum = np.fft.fft2(field) / field.size
row, column = np.unravel_index(np.argmax(abs(spectrum)), spectrum.shape)
recovered = 2 * np.pi * np.array([frequency[column], frequency[row]])

# %% CHECK AND INTERPRET
assert np.allclose(recovered, [kx, kz])
print("Wavelength (um):", 2 * np.pi / np.hypot(kx, kz), "; recovered vector:", recovered)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    ax.imshow(abs(np.fft.fftshift(spectrum)), origin='lower')
    ax.set(xlabel='Shifted x-frequency bin', ylabel='Shifted z-frequency bin', title='One Fourier peak')

    # %% NUMERICAL SUMMARY
    print('1.3 - Find a plane-wave direction using a two-dimensional FFT')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
