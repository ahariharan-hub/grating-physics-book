# -*- coding: utf-8 -*-
"""7.3, worked example 1: Implement a Gaussian beam in its own rotated coordinate frame.

Companion topic: From ideal orders to finite beams.
Prescribed paraxial Gaussian order beams; wavelength 1 um, pitch 4 um, outgoing -z. No material scattering or recombination optics are solved.
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
from wave_optics.gaussian_order_beams import GaussianOrderConfiguration, GaussianOrderExperiment

# %% INPUT PARAMETERS
wavelength = 1.0
period = 4.0
waist = 6.0
order = 1
depth = 12.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
x = np.linspace(-20, 20, 801)
points = np.column_stack([x, np.zeros_like(x), np.full_like(x, -depth)])
sine = order * wavelength / period
cosine = np.sqrt(1 - sine**2)

# %% NUMERICAL CALCULATION
direction = np.array([sine, 0.0, -cosine])
transverse_direction = np.array([-cosine, 0.0, -sine])
longitudinal = points @ direction
transverse = points @ transverse_direction
zR = np.pi * waist**2 / wavelength
width = waist * np.sqrt(1 + (longitudinal / zR)**2)
inverse_curvature = longitudinal / (longitudinal**2 + zR**2)
k = 2 * np.pi / wavelength
phase = k * longitudinal + k * transverse**2 * inverse_curvature / 2 - np.arctan(longitudinal / zR)
field = waist / width * np.exp(-(transverse / width)**2 + 1j * phase)
model = GaussianOrderExperiment(GaussianOrderConfiguration(wavelength=wavelength, period=period, waist=waist))
backend = model.field(points, order)

# %% CHECK AND INTERPRET
assert np.allclose(field, backend)
print("Maximum explicit/backend Gaussian difference:", np.max(abs(field - backend)))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(x, abs(field)**2, label='Explicit rotated Gaussian')
    ax.plot(x, abs(backend)**2, '--', label='Backend check')
    ax.set(xlabel='Common detector x (um)', ylabel='Relative intensity')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('7.3 - Implement a Gaussian beam in its own rotated coordinate frame')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
