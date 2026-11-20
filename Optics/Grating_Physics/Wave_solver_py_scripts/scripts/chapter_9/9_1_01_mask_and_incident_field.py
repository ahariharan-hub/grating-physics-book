# -*- coding: utf-8 -*-
"""9.1, worked example 1: Construct a complex exit field from a prescribed mask and illumination.

Companion topic: Build the exit field from a mask and illumination.
Lengths um. A thin prescribed transmission model at one plane, not a scattering solve.
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
period = 4.0
clear_fraction = 0.4
incident_frequency = 0.12
samples = 1024
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = -period / 2 + np.arange(samples) * period / samples

# %% NUMERICAL CALCULATION
transmission = (abs(x) < clear_fraction * period / 2).astype(float)
incident = np.exp(2j * np.pi * incident_frequency * x)
exit_field = transmission * incident
amplitude = abs(exit_field)
phase = np.where(amplitude > 1e-8, np.angle(exit_field), np.nan)

# %% CHECK AND INTERPRET
assert np.allclose(amplitude, transmission)
print("Phase at zero amplitude is undefined and is omitted from the phase plot.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, amplitude)
    axes[0].set(xlabel='x (um)', ylabel='Exit amplitude')
    axes[1].plot(x, phase)
    axes[1].set(xlabel='x (um)', ylabel='Exit phase (rad)')

    # %% NUMERICAL SUMMARY
    print('9.1 - Construct a complex exit field from a prescribed mask and illumination')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
