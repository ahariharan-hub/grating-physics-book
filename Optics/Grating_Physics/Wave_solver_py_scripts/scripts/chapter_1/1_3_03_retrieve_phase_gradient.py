# -*- coding: utf-8 -*-
"""1.3, worked example 3: Recover a wavevector from sampled complex phase.

Companion topic: Phase gradients and propagation direction.
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
angle_degrees = 30.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-4, 4, 161)
z = np.linspace(0, 8, 161)
xx, zz = np.meshgrid(x, z)
k = 2 * np.pi / wavelength

# %% NUMERICAL CALCULATION
kx = k * np.sin(np.deg2rad(angle_degrees))
kz = k * np.cos(np.deg2rad(angle_degrees))
field = np.exp(1j * (kx * xx + kz * zz))
phase = np.unwrap(np.unwrap(np.angle(field), axis=1), axis=0)
gradient_z, gradient_x = np.gradient(phase, z, x)

# %% CHECK AND INTERPRET
assert np.allclose(gradient_x, kx)
assert np.allclose(gradient_z, kz)
print("Recovered (kx,kz):", gradient_x.mean(), gradient_z.mean())

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].imshow(np.angle(field), origin='lower', extent=[-4, 4, 0, 8], cmap='twilight')
    axes[0].set(xlabel='x (um)', ylabel='z (um)', title='Wrapped phase')
    axes[1].imshow(phase, origin='lower', extent=[-4, 4, 0, 8])
    axes[1].set(xlabel='x (um)', ylabel='z (um)', title='Unwrapped phase')

    # %% NUMERICAL SUMMARY
    print('1.3 - Recover a wavevector from sampled complex phase')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
