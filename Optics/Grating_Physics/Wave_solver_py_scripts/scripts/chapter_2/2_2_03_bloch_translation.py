# -*- coding: utf-8 -*-
"""2.2, worked example 3: Verify the Bloch translation law for illuminated openings.

Companion topic: Surface-field translation.
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
period = 3.0
clear_fraction = 0.4
incidence_degrees = 17.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-1.45, 1.45, 501)
kx = 2 * np.pi / wavelength * np.sin(np.deg2rad(incidence_degrees))

# %% NUMERICAL CALCULATION
mask = (abs((x / period + 0.5) % 1 - 0.5) < clear_fraction / 2).astype(float)
shifted_mask = (abs(((x + period) / period + 0.5) % 1 - 0.5) < clear_fraction / 2).astype(float)
field = mask * np.exp(1j * kx * x)
shifted = shifted_mask * np.exp(1j * kx * (x + period))
reduced = field * np.exp(-1j * kx * x)

# %% CHECK AND INTERPRET
assert np.allclose(shifted, field * np.exp(1j * kx * period))
assert np.allclose(reduced, mask)
print("Removing the incident carrier leaves a periodic unit-cell function.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, field.real, label='Illuminated real field')
    ax.plot(x, reduced.real, '--', label='Carrier removed')
    ax.set(xlabel='x (um)', ylabel='Real field')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('2.2 - Verify the Bloch translation law for illuminated openings')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
