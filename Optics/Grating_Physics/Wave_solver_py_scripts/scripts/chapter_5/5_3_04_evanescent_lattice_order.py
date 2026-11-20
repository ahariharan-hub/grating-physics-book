# -*- coding: utf-8 -*-
"""5.3, worked example 4: Propagate one evanescent two-dimensional order.

Companion topic: Follow an evanescent order into the output medium.
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
pitch_x = 4.0
pitch_y = 5.0
order = np.array([1, 1])
incident_direction = np.array([0.3, 0.4])
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
s = np.linspace(0, 3, 301)

# %% NUMERICAL CALCULATION
transverse = k * incident_direction + 2 * np.pi * order / np.array([pitch_x, pitch_y])
q = np.sqrt(complex(k**2 - np.dot(transverse, transverse)))
field_on_axis = np.exp(1j * q * s)
analytic_envelope = np.exp(-q.imag * s)

# %% CHECK AND INTERPRET
assert q.imag > 0
assert np.allclose(abs(field_on_axis), analytic_envelope)
print("This (m,n) order has no real outgoing ray; its surface amplitude is prescribed.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(s, abs(field_on_axis), label='Amplitude')
    ax.plot(s, abs(field_on_axis)**2, label='Relative intensity')
    ax.set(xlabel='Downstream s=-z (um)', ylabel='Relative value')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.3 - Propagate one evanescent two-dimensional order')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
