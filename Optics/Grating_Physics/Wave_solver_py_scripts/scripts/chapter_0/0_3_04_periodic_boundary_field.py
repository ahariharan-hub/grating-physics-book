# -*- coding: utf-8 -*-
"""0.3, worked example 4: Prescribe an illuminated grating boundary and test its translation.

Companion topic: How does this connect to a grating?.
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
clear_fraction = 0.4
incidence_degrees = 20.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-6.0, 6.0, 1201)
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
local = (x / period + 0.5) % 1 - 0.5
mask = (abs(local) < clear_fraction / 2).astype(float)
field = mask * np.exp(1j * kx * x)
translated = mask * np.exp(1j * kx * (x + period))
translation_factor = np.exp(1j * kx * period)

# %% CHECK AND INTERPRET
assert np.allclose(translated, translation_factor * field)
print('The mask is periodic; the illuminated complex field is generally Bloch periodic.')

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, mask, '--', label='Binary amplitude mask')
    ax.plot(x, field.real, label='Real boundary field')
    ax.plot(x, field.imag, label='Imaginary boundary field')
    ax.set(xlabel='x (um)', ylabel='Field or amplitude transmission')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('0.3 - Prescribe an illuminated grating boundary and test its translation')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
