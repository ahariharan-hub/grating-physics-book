# -*- coding: utf-8 -*-
"""0.4, worked example 3: Propagate a sampled slit by the angular-spectrum method.

Companion topic: Reconstruct the propagated field.
Prescribed 1D slit; no material scattering. Medium wavelength, um; downstream s=-z; exp(-i omega t).
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
width = 3.0
distance = 8.0
window = 48.0
samples = 1024
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = window / samples
x = (np.arange(samples) - samples // 2) * dx
fx = np.fft.fftfreq(samples, dx)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
boundary = (abs(x) <= width / 2).astype(complex)
spectrum = np.fft.fft(boundary)
q = k * np.sqrt((1 - (wavelength * fx)**2).astype(complex))
transfer = np.exp(1j * q * distance)
field = np.fft.ifft(spectrum * transfer)
at_zero_distance = np.fft.ifft(spectrum)

# %% CHECK AND INTERPRET
assert np.allclose(at_zero_distance, boundary)
assert np.all(np.isfinite(field))
print('dx =', dx, 'um. Increase window and samples independently to check FFT artifacts.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(boundary)**2, '--', label='Prescribed boundary intensity')
    ax.plot(x, abs(field)**2, label='Propagated intensity')
    ax.set(xlabel='x (um)', ylabel='Relative intensity', xlim=(-12, 12))
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.4 - Propagate a sampled slit by the angular-spectrum method')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
