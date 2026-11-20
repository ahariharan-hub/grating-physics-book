# -*- coding: utf-8 -*-
"""0.1, worked example 2: Recover a real field from a complex phasor.

Companion topic: Separate the monochromatic time dependence.
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
amplitude = 1.3
phase_degrees = 35.0
period = 2.0  # fs
show_plot = True

# %% SETUP: coordinates, sampling and constants
time = np.linspace(0.0, 3 * period, 1201)
omega = 2 * np.pi / period
phase = np.deg2rad(phase_degrees)

# %% NUMERICAL CALCULATION
phasor = amplitude * np.exp(1j * phase)
real_field = np.real(phasor * np.exp(-1j * omega * time))
direct_cosine = amplitude * np.cos(phase - omega * time)

# %% CHECK AND INTERPRET
assert np.allclose(real_field, direct_cosine)
print('Phasor magnitude:', abs(phasor), '; phase (rad):', np.angle(phasor))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(time / period, real_field, label='Re[U exp(-i omega t)]')
    ax.plot(time / period, direct_cosine, '--', label='Direct cosine')
    ax.set(xlabel='Time / period', ylabel='Real scalar field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.1 - Recover a real field from a complex phasor')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
