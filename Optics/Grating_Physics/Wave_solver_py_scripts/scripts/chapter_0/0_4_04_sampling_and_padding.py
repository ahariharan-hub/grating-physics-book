# -*- coding: utf-8 -*-
"""0.4, worked example 4: Separate FFT resolution from window-padding effects.

Companion topic: How does ASM relate to numerical sampling?.
Prescribed Gaussian boundary; 1D scalar ASM, downstream s=-z; medium wavelength and lengths um.
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
distance = 25.0
grids = [(24.0, 256), (48.0, 512), (48.0, 1024)]
show_plot = True

# %% SETUP: coordinates, sampling and constants
detector_x = np.linspace(-8.0, 8.0, 301)
profiles = []
roundtrip_errors = []

# %% NUMERICAL CALCULATION
for window, count in grids:
    dx = window / count
    x = (np.arange(count) - count // 2) * dx
    u0 = np.exp(-(x / waist)**2)
    frequencies = np.fft.fftfreq(count, dx)
    q = 2 * np.pi / wavelength * np.sqrt((1 - (wavelength * frequencies)**2).astype(complex))
    spectrum = np.fft.fft(u0)
    field = np.fft.ifft(spectrum * np.exp(1j * q * distance))
    profiles.append(np.interp(detector_x, x, abs(field)**2))
    roundtrip_errors.append(np.max(abs(np.fft.ifft(spectrum) - u0)))

# %% CHECK AND INTERPRET
assert max(roundtrip_errors) < 1e-12
print('Profile differences against the finest padded case:',
      [np.max(abs(p - profiles[-1])) for p in profiles[:-1]])
print('FFT roundtrip accuracy alone does not establish propagation accuracy.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for (window, count), profile in zip(grids, profiles):
        ax.plot(detector_x, profile, label=f'Window {window:g} um; N={count}')
    ax.set(xlabel='Detector x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.4 - Separate FFT resolution from window-padding effects')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
