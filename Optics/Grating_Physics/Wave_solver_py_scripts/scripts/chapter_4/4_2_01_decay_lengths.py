# -*- coding: utf-8 -*-
"""4.2, worked example 1: Distinguish amplitude decay length from intensity decay length.

Companion topic: How far does the field extend?.
Medium wavelength and lengths um. Downstream s=-z; U=exp(i kx x+i q s); positive-imaginary q decays.
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
wavelength = 2.0
transverse_ratio = 1.5
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
kappa = k * np.sqrt(transverse_ratio**2 - 1)
amplitude_length = 1 / kappa
s = np.linspace(0, 4 * amplitude_length, 401)

# %% NUMERICAL CALCULATION
amplitude = np.exp(-kappa * s)
intensity = abs(amplitude)**2
intensity_length = 1 / (2 * kappa)

# %% CHECK AND INTERPRET
assert np.isclose(np.exp(-kappa * amplitude_length), np.exp(-1))
assert np.isclose(np.exp(-2 * kappa * intensity_length), np.exp(-1))
print("Amplitude/intensity decay lengths (um):", amplitude_length, intensity_length)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(s / amplitude_length, amplitude, label='Amplitude')
    ax.plot(s / amplitude_length, intensity, label='Intensity')
    ax.set(xlabel='Depth / amplitude decay length', ylabel='Relative value')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('4.2 - Distinguish amplitude decay length from intensity decay length')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
