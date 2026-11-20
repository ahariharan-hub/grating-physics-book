# -*- coding: utf-8 -*-
"""1.1, worked example 4: Estimate oscillation frequency with a sampled Fourier transform.

Companion topic: Period, frequency and numerical sampling.
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
frequency = 0.5  # inverse fs
sample_interval = 0.05
samples = 800
show_plot = True

# %% SETUP: coordinates, sampling and constants
time = np.arange(samples) * sample_interval
frequencies = np.fft.rfftfreq(samples, sample_interval)

# %% NUMERICAL CALCULATION
field = np.cos(2 * np.pi * frequency * time + 0.3)
spectrum = np.fft.rfft(field)
peak = np.argmax(abs(spectrum[1:])) + 1
estimated_frequency = frequencies[peak]

# %% CHECK AND INTERPRET
assert abs(estimated_frequency - frequency) <= 1 / (samples * sample_interval)
print("Estimated period (fs):", 1 / estimated_frequency)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(frequencies, abs(spectrum) / samples)
    ax.set(xlabel='Frequency (inverse fs)', ylabel='DFT magnitude / N', xlim=(0, 2))

    # %% NUMERICAL SUMMARY
    print('1.1 - Estimate oscillation frequency with a sampled Fourier transform')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
