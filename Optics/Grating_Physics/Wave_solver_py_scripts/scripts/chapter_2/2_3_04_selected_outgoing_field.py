# -*- coding: utf-8 -*-
"""2.3, worked example 4: Construct a selected outgoing wave without assigning an efficiency.

Companion topic: Build one outgoing wavevector.
Prescribed scalar component, medium wavelength and lengths um; downstream s=-z; exp(-i omega t).
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
order = 1
incidence_degrees = 10.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-4, 4, 241)
s = np.linspace(0, 8, 241)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
kx = k * np.sin(np.deg2rad(incidence_degrees)) + order * 2 * np.pi / period
q = np.sqrt(complex(k**2 - kx**2))
field = np.exp(1j * kx * x[None, :] + 1j * q * s[:, None])

# %% CHECK AND INTERPRET
assert np.isclose(kx**2 + q**2, k**2)
assert np.max(abs(field)) <= 1 + 1e-12
print("Unit surface amplitude is prescribed; its physical excitation is not solved.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    image = ax.imshow(field.real, origin='lower', extent=[x[0], x[-1], s[0], s[-1]], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set(xlabel='x (um)', ylabel='s=-z (um)')
    figure.colorbar(image, ax=ax, label='Real field')

    # %% NUMERICAL SUMMARY
    print('2.3 - Construct a selected outgoing wave without assigning an efficiency')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
