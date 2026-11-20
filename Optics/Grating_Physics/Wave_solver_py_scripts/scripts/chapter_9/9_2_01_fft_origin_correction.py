# -*- coding: utf-8 -*-
"""9.2, worked example 1: Recover complex Fourier coefficients with the sampling-origin phase correction.

Companion topic: Sample one period and recover its coefficients.
Coordinates nm and Fourier frequencies cycles/nm; prescribed finite periodic complex field. No scattering or propagation is solved.
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
samples = 32
start_fraction = -0.5
orders = np.array([-3, 0, 2, 5])
coefficients = np.array([0.25 * np.exp(0.6j), 0.8 * np.exp(-0.3j), 0.4 * np.exp(0.9j), 0.2 * np.exp(-0.7j)])
show_plot = True

# %% SETUP: coordinates, sampling and constants
dx = period / samples
x0 = start_fraction * period
x = x0 + np.arange(samples) * dx
frequencies = np.fft.fftfreq(samples, dx)

# %% NUMERICAL CALCULATION
field = np.exp(2j * np.pi * x[:, None] * orders / period) @ coefficients
raw = np.fft.fft(field) / samples
corrected = raw * np.exp(-2j * np.pi * frequencies * x0)
recovered = corrected[orders % samples]

# %% CHECK AND INTERPRET
assert np.allclose(recovered, coefficients)
print("Raw/corrected coefficient errors:", np.max(abs(raw[orders % samples] - coefficients)), np.max(abs(recovered - coefficients)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(orders, np.angle(coefficients), 'o', label='Specified phase')
    ax.plot(orders, np.angle(raw[orders % samples]), 'x', label='Uncorrected FFT phase')
    ax.plot(orders, np.angle(recovered), '+', markersize=10, label='Corrected FFT phase')
    ax.set(xlabel='Order m', ylabel='Coefficient phase (rad)')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.2 - Recover complex Fourier coefficients with the sampling-origin phase correction')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
