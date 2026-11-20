# -*- coding: utf-8 -*-
"""9.2, worked example 3: Track FFT-bin changes when the window contains several periods.

Companion topic: Sample more than one complete period.
Coordinates nm; sample spacing fixed while the integer-period window grows. This is not zero padding of one sampled period.
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
period = 1000.0
samples_per_period = 32
window_periods = [1, 2, 4]
orders = np.array([-3, 0, 2, 5])
coefficients = np.array([0.25j, 0.8, 0.4 * np.exp(0.9j), 0.2])
show_plot = True

# %% SETUP: coordinates, sampling and constants
spectra = []

# %% NUMERICAL CALCULATION
for periods in window_periods:
    samples = samples_per_period * periods
    dx = period / samples_per_period
    x = np.arange(samples) * dx
    field = np.exp(2j * np.pi * x[:, None] * orders / period) @ coefficients
    spectrum = np.fft.fft(field) / samples
    frequencies = np.fft.fftfreq(samples, dx)
    recovered = spectrum[(orders * periods) % samples]
    assert np.allclose(recovered, coefficients)
    spectra.append((frequencies, spectrum))

# %% CHECK AND INTERPRET
print("The physical frequencies stay fixed; their DFT-bin indices scale with the number of periods.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for periods, (frequency, spectrum) in zip(window_periods, spectra):
        ax.plot(np.fft.fftshift(frequency), abs(np.fft.fftshift(spectrum)), 'o-', label=f'{periods} periods')
    ax.set(xlabel='Frequency (cycles/nm)', ylabel='Normalized DFT magnitude', xlim=(-0.006, 0.008))
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.2 - Track FFT-bin changes when the window contains several periods')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
