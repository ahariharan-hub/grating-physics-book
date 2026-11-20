# -*- coding: utf-8 -*-
"""6.2, worked example 2: Measure spatial phase while detector intensity remains uniform.

Companion topic: One order across a detector.
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
wavelength = 2.0
angle_degrees = 35.0
detector_depth = 3.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-6, 6, 601)
k = 2 * np.pi / wavelength
kx = k * np.sin(np.deg2rad(angle_degrees))
q = k * np.cos(np.deg2rad(angle_degrees))

# %% NUMERICAL CALCULATION
field = np.exp(1j * (kx * x + q * detector_depth))
phase = np.unwrap(np.angle(field))
phase_slope = np.gradient(phase, x)
intensity = abs(field)**2

# %% CHECK AND INTERPRET
assert np.allclose(phase_slope, kx)
assert np.allclose(intensity, 1)
print("A spatial phase ramp in one plane wave does not create intensity fringes.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(x, phase)
    axes[0].set(xlabel='x (um)', ylabel='Unwrapped phase (rad)')
    axes[1].plot(x, intensity)
    axes[1].set(xlabel='x (um)', ylabel='Uniform relative intensity', ylim=(0, 1.2))

    # %% NUMERICAL SUMMARY
    print('6.2 - Measure spatial phase while detector intensity remains uniform')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
