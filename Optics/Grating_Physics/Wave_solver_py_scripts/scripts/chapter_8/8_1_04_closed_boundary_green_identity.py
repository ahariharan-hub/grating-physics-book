# -*- coding: utf-8 -*-
"""8.1, worked example 4: Reconstruct a source-free interior field from a closed boundary.

Companion topic: How can the boundary reconstruct the field?.
3D outgoing Green function, medium wavelength and lengths um. The point source is outside the integration sphere, so its interior is source-free.
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
sphere_radius = 2.0
source = np.array([0.0, 0.0, -4.0])
observation = np.array([0.2, 0.0, 0.3])
polar_order = 32
azimuth_samples = 96
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
mu, mu_weights = np.polynomial.legendre.leggauss(polar_order)
phi = 2 * np.pi * np.arange(azimuth_samples) / azimuth_samples
mm, pp = np.meshgrid(mu, phi, indexing="ij")

# %% NUMERICAL CALCULATION
normals = np.stack([np.sqrt(1 - mm**2) * np.cos(pp), np.sqrt(1 - mm**2) * np.sin(pp), mm], axis=-1)
boundary = sphere_radius * normals
from_source = boundary - source
to_observation = boundary - observation
r_source = np.linalg.norm(from_source, axis=-1)
r_green = np.linalg.norm(to_observation, axis=-1)
field = np.exp(1j * k * r_source) / (4 * np.pi * r_source)
green = np.exp(1j * k * r_green) / (4 * np.pi * r_green)
derivative_field = field * (1j * k - 1 / r_source) * np.sum(from_source * normals, axis=-1) / r_source
derivative_green = green * (1j * k - 1 / r_green) * np.sum(to_observation * normals, axis=-1) / r_green
area_weights = sphere_radius**2 * mu_weights[:, None] * 2 * np.pi / azimuth_samples
contributions = (green * derivative_field - field * derivative_green) * area_weights
reconstructed = contributions.sum()
distance = np.linalg.norm(observation - source)
reference = np.exp(1j * k * distance) / (4 * np.pi * distance)

# %% CHECK AND INTERPRET
assert np.linalg.norm(source) > sphere_radius > np.linalg.norm(observation)
assert abs(reconstructed - reference) / abs(reference) < 1e-6
print("Closed-boundary relative complex-field error:", abs(reconstructed - reference) / abs(reference))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    path = np.r_[0j, np.cumsum(contributions.ravel())]
    ax.plot(path.real, path.imag, label='Cumulative boundary sum')
    ax.plot(reference.real, reference.imag, 'rx', markersize=9, label='Exact interior field')
    ax.set(xlabel='Real field', ylabel='Imaginary field', aspect='equal')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('8.1 - Reconstruct a source-free interior field from a closed boundary')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
