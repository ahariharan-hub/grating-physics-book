# -*- coding: utf-8 -*-
"""6.1, worked example 4: Show that phase-distinct fields can have equal recorded intensity.

Companion topic: Read the measurement in reverse.
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
amplitude = 1.0
phases_degrees = [0.0, 45.0, 90.0, 180.0]
show_plot = True

# %% SETUP: coordinates, sampling and constants
time = np.arange(1000) / 1000
phases = np.deg2rad(phases_degrees)

# %% NUMERICAL CALCULATION
phasors = amplitude * np.exp(1j * phases)
fields = np.real(phasors[:, None] * np.exp(-2j * np.pi * time))
mean_squares = np.mean(fields**2, axis=1)
relative_intensities = abs(phasors)**2

# %% CHECK AND INTERPRET
assert np.allclose(mean_squares, amplitude**2 / 2)
assert np.allclose(relative_intensities, amplitude**2)
print("An intensity measurement alone does not recover the optical phase.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for phase, field in zip(phases_degrees, fields):
        axes[0].plot(time, field, label=f'{phase:g} deg')
    axes[0].set(xlabel='t / T', ylabel='Real field')
    axes[0].legend()
    axes[1].bar(np.arange(len(phases)), relative_intensities)
    axes[1].set(xlabel='Phase-case index', ylabel='Identical relative intensity')

    # %% NUMERICAL SUMMARY
    print('6.1 - Show that phase-distinct fields can have equal recorded intensity')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
