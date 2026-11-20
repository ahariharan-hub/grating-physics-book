# -*- coding: utf-8 -*-
"""8.2, worked example 4: Compare a direct RS aperture integral with FFT angular-spectrum propagation.

Companion topic: Spatial patches and Fourier components describe the same propagation.
Prescribed Gaussian boundary in two transverse dimensions; full scalar propagation. Medium wavelength and lengths um; downstream s=-z.
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
distance = 4.0
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
q = k * np.sqrt((1 - wavelength**2 * (fx**2 + fy**2)).astype(complex))
asm = np.fft.ifft2(np.fft.fft2(boundary) * np.exp(1j * q * distance))
rs = []
for x_observer in x[indices]:
    radius = np.sqrt((x_observer - xx)**2 + yy**2 + distance**2)
    kernel = distance * np.exp(1j * k * radius) * (1 - 1j * k * radius) / (2 * np.pi * radius**3)
    rs.append(np.sum(boundary * kernel) * dx**2)
rs = np.array(rs)
asm_cut = asm[samples // 2, indices]
relative_difference = np.linalg.norm(rs - asm_cut) / np.linalg.norm(rs)

# %% CHECK AND INTERPRET
assert relative_difference < 0.02
print("Relative RS/ASM complex-field difference:", relative_difference)
print("Finite integration domain and periodic FFT boundaries need independent checks.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x[indices], rs.real, 'o', label='Direct RS real field')
    ax.plot(x[indices], asm_cut.real, '--', label='FFT ASM real field')
    ax.plot(x[indices], rs.imag, 's', label='Direct RS imaginary field')
    ax.plot(x[indices], asm_cut.imag, ':', label='FFT ASM imaginary field')
    ax.set(xlabel='Detector x (um)', ylabel='Complex field component')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.2 - Compare a direct RS aperture integral with FFT angular-spectrum propagation')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
