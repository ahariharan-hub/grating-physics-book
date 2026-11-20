# -*- coding: utf-8 -*-
"""9.2, worked example 2: Demonstrate that a perfect FFT roundtrip can hide an aliased field.

Companion topic: What if we use only eight points?.
Coordinates nm; a prescribed complex Fourier sum. The DFT interpolant is not necessarily the true underlying continuous field.
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
samples = 8
orders = np.array([-3, 0, 2, 5])
coefficients = np.array([0.25 * np.exp(0.6j), 0.8 * np.exp(-0.3j), 0.4 * np.exp(0.9j), 0.2 * np.exp(-0.7j)])
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.arange(samples) * period / samples
fine_x = np.linspace(0, period, 1001)
frequencies = np.fft.fftfreq(samples, period / samples)

# %% NUMERICAL CALCULATION
sampled = np.exp(2j * np.pi * x[:, None] * orders / period) @ coefficients
fft_coefficients = np.fft.fft(sampled) / samples
reconstructed_samples = np.fft.ifft(fft_coefficients * samples)
interpolant = np.exp(2j * np.pi * fine_x[:, None] * frequencies) @ fft_coefficients
reference = np.exp(2j * np.pi * fine_x[:, None] * orders / period) @ coefficients
collision = fft_coefficients[(-3) % samples]

# %% CHECK AND INTERPRET
assert np.allclose(reconstructed_samples, sampled)
if samples == 8 and np.array_equal(orders, [-3, 0, 2, 5]):
    assert np.allclose(collision, coefficients[0] + coefficients[3])
    assert np.linalg.norm(interpolant - reference) > 0.1
if samples > 2 * np.max(abs(orders)):
    assert np.allclose(interpolant, reference)
print('Continuous-field relative error:', np.linalg.norm(interpolant - reference) / np.linalg.norm(reference))
print('Orders -3 and +5 collide at N=8; increasing N can resolve them without changing the field.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(fine_x, reference.real, label='Known continuous field')
    ax.plot(fine_x, interpolant.real, '--', label='Aliased DFT interpolant')
    ax.plot(x, sampled.real, 'o', label='Identical coarse samples')
    ax.set(xlabel='x (nm)', ylabel='Real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('9.2 - Demonstrate that a perfect FFT roundtrip can hide an aliased field')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
