# -*- coding: utf-8 -*-
"""6.1, worked example 2: Verify the square-law relation with a fixed reference.

Companion topic: Convert mean square into relative recorded intensity.
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
amplitudes = np.linspace(0, 2, 101)
reference_amplitude = 1.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
time = np.arange(1000) / 1000

# %% NUMERICAL CALCULATION
fields = amplitudes[:, None] * np.cos(2 * np.pi * time)
mean_squares = np.mean(fields**2, axis=1)
reference_mean_square = reference_amplitude**2 / 2
relative_intensity = mean_squares / reference_mean_square

# %% CHECK AND INTERPRET
assert np.allclose(relative_intensity, (amplitudes / reference_amplitude)**2)
print("Doubling field amplitude quadruples relative intensity when the reference stays fixed.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(amplitudes, relative_intensity)
    ax.set(xlabel='Field amplitude / fixed unit', ylabel='Relative intensity')

    # %% NUMERICAL SUMMARY
    print('6.1 - Verify the square-law relation with a fixed reference')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
