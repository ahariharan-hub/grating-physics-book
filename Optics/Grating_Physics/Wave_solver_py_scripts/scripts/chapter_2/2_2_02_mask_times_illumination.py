# -*- coding: utf-8 -*-
"""2.2, worked example 2: Multiply a binary transmission by a tilted complex field.

Companion topic: What does the grating do at the surface?.
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
period = 2.0
width = 1.0
incidence_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-4, 4, 1601)
local = (x + period / 2) % period - period / 2
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
mask = (abs(local) < width / 2).astype(float)
incident = np.exp(1j * kx * x)
exit_field = mask * incident

# %% CHECK AND INTERPRET
assert np.allclose(abs(exit_field), mask)
assert np.all(exit_field[mask == 0] == 0)
print("Illumination changes phase inside openings; it does not alter the aperture geometry.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, mask, ':', label='Transmission')
    ax.plot(x, exit_field.real, label='Exit real part')
    ax.plot(x, exit_field.imag, label='Exit imaginary part')
    ax.set(xlabel='x (um)', ylabel='Field or transmission')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.2 - Multiply a binary transmission by a tilted complex field')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
