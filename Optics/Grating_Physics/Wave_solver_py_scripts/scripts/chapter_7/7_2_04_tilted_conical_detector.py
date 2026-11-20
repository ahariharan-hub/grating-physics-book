# -*- coding: utf-8 -*-
"""7.2, worked example 4: Project conical-order interference onto a tilted detector.

Companion topic: Conical diffraction interference.
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
alpha_incident = 0.2
beta_incident = 0.3
detector_tilt_degrees = 25.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
orders = np.array([0, 1])
coordinate = np.linspace(-4, 4, 161)
u, v = np.meshgrid(coordinate, coordinate)
tilt = np.deg2rad(detector_tilt_degrees)

# %% NUMERICAL CALCULATION
alpha = alpha_incident + orders * wavelength / period
vectors = k * np.column_stack([alpha, np.full(2, beta_incident), -np.sqrt(1 - alpha**2 - beta_incident**2)])
e_u = np.array([1.0, 0.0, 0.0])
e_v = np.array([0.0, np.cos(tilt), np.sin(tilt)])
points = np.array([0.0, 0.0, -5.0]) + u[..., None] * e_u + v[..., None] * e_v
fields = np.exp(1j * (points @ vectors.T))
intensity = abs(fields.sum(axis=-1))**2
delta = vectors[1] - vectors[0]
expanded = 2 + 2 * np.cos(points @ delta)

# %% CHECK AND INTERPRET
assert np.allclose(intensity, expanded)
print("Detector-plane phase gradients (rad/um):", delta @ e_u, delta @ e_v)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    image = ax.imshow(intensity, origin='lower', extent=[-4, 4, -4, 4])
    ax.set(xlabel='Detector u (um)', ylabel='Detector v (um)')
    figure.colorbar(image, ax=ax, label='Relative intensity')

    # %% NUMERICAL SUMMARY
    print('7.2 - Project conical-order interference onto a tilted detector')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
