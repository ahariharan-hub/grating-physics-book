# -*- coding: utf-8 -*-
"""2.1, worked example 3: Combine opening width with the finite-array phase sum.

Companion topic: From an opening to a finite grating.
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
wavelength = 1.0
period = 4.0
width = 2.0
openings = 5
show_plot = True

# %% SETUP: coordinates, sampling and constants
angles = np.linspace(-60, 60, 1201)
positions = (np.arange(openings) - (openings - 1) / 2) * period
sine = np.sin(np.deg2rad(angles))

# %% NUMERICAL CALCULATION
envelope = np.sinc(width * sine / wavelength)
array = np.exp(-2j * np.pi / wavelength * sine[:, None] * positions).mean(axis=1)
intensity = abs(envelope * array)**2
first_envelope_zero = np.rad2deg(np.arcsin(wavelength / width))

# %% CHECK AND INTERPRET
assert 0 < width <= period
assert abs(np.sinc(width * np.sin(np.deg2rad(first_envelope_zero)) / wavelength)) < 1e-12
print("First single-slit zero (deg):", first_envelope_zero)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(angles, intensity, label='Slit envelope times array')
    ax.plot(angles, envelope**2, '--', label='Single-slit envelope')
    ax.set(xlabel='Angle (deg)', ylabel='Peak-normalized intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.1 - Combine opening width with the finite-array phase sum')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
