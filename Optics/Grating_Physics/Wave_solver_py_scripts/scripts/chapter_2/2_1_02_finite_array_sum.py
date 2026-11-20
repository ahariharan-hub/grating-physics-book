# -*- coding: utf-8 -*-
"""2.1, worked example 2: Sum equally driven openings and recover the finite-array factor.

Companion topic: Repeat the opening to make a grating.
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
openings = 7
show_plot = True

# %% SETUP: coordinates, sampling and constants
angles = np.linspace(-60, 60, 1801)
positions = (np.arange(openings) - (openings - 1) / 2) * period
phase_step = 2 * np.pi * period / wavelength * np.sin(np.deg2rad(angles))

# %% NUMERICAL CALCULATION
amplitude = np.exp(-1j * phase_step[:, None] * positions / period).mean(axis=1)
denominator = openings * np.sin(phase_step / 2)
analytic = np.divide(np.sin(openings * phase_step / 2), denominator, out=np.ones_like(phase_step), where=abs(denominator) > 1e-10)

# %% CHECK AND INTERPRET
assert np.allclose(abs(amplitude)**2, analytic**2, atol=1e-10)
print("Finite-array peaks have finite width; this model omits the single-opening envelope.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(angles, abs(amplitude)**2)
    ax.set(xlabel='Angle (deg)', ylabel='Array factor magnitude squared / N squared')

    # %% NUMERICAL SUMMARY
    print('2.1 - Sum equally driven openings and recover the finite-array factor')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
