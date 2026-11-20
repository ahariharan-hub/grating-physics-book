# -*- coding: utf-8 -*-
"""1.1, worked example 1: Sample one optical oscillation in time.

Companion topic: Oscillation and phase.
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
phase_degrees = 30.0
period = 2.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
time = np.linspace(0, 2 * period, 801)
phase = np.deg2rad(phase_degrees)

# %% NUMERICAL CALCULATION
phasor = amplitude * np.exp(1j * phase)
field = np.real(phasor * np.exp(-2j * np.pi * time / period))
quarter_period = np.real(phasor * np.exp(-2j * np.pi * 0.25))

# %% CHECK AND INTERPRET
assert np.isclose(quarter_period, amplitude * np.sin(phase))
print("Quarter-period field:", quarter_period)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(time / period, field)
    ax.set(xlabel='t / T', ylabel='Real field', title='One observation point')

    # %% NUMERICAL SUMMARY
    print('1.1 - Sample one optical oscillation in time')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
