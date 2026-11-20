# -*- coding: utf-8 -*-
"""8.2, worked example 1: Integrate a rectangular aperture using explicit two-dimensional Simpson weights.

Companion topic: Try the RS I patch sum.
Medium wavelength and lengths um; prescribed scalar aperture; downstream s=-z. This propagates a field and does not solve screen scattering.
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
aperture_width_x = 3.0
aperture_width_y = 2.0
distance = 8.0
aperture_points = 41
observation_points = 41
observation_half_width = 6.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
assert aperture_points % 2 == 1
k = 2 * np.pi / wavelength
xi = np.linspace(-aperture_width_x / 2, aperture_width_x / 2, aperture_points)
eta = np.linspace(-aperture_width_y / 2, aperture_width_y / 2, aperture_points)
xx_source, yy_source = np.meshgrid(xi, eta)
dx = xi[1] - xi[0]
dy = eta[1] - eta[0]
simpson_weights = np.ones(aperture_points)
simpson_weights[1:-1:2] = 4
simpson_weights[2:-1:2] = 2
area_weights = np.outer(simpson_weights, simpson_weights) * dx * dy / 9
x = np.linspace(-observation_half_width, observation_half_width, observation_points)
y = x.copy()
field = np.zeros((observation_points, observation_points), dtype=complex)

# %% NUMERICAL CALCULATION
# U0=1 inside the rectangular aperture, zero elsewhere.
# Full 3D RS I kernel: s exp(ikR)(1-ikR)/(2 pi R^3).
for row, y_observer in enumerate(y):
    for column, x_observer in enumerate(x):
        radius = np.sqrt((x_observer - xx_source)**2 + (y_observer - yy_source)**2 + distance**2)
        kernel = distance * np.exp(1j * k * radius) * (1 - 1j * k * radius) / (2 * np.pi * radius**3)
        field[row, column] = np.sum(kernel * area_weights)
intensity = abs(field)**2

# %% CHECK AND INTERPRET
assert np.isclose(area_weights.sum(), aperture_width_x * aperture_width_y)
assert np.allclose(intensity, intensity[::-1, :], atol=1e-12)
assert np.allclose(intensity, intensity[:, ::-1], atol=1e-12)
print("On-axis complex field:", field[observation_points // 2, observation_points // 2])
print("Refine aperture and detector grids separately; no per-setting intensity normalization is applied.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    image = axes[0].imshow(intensity, origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
    axes[0].set(xlabel='Detector x (um)', ylabel='Detector y (um)', title='Rectangular-aperture RS I intensity')
    figure.colorbar(image, ax=axes[0], label='Relative intensity')
    axes[1].plot(x, intensity[observation_points // 2], label='x section')
    axes[1].plot(y, intensity[:, observation_points // 2], label='y section')
    axes[1].set(xlabel='Detector coordinate (um)', ylabel='Relative intensity')
    axes[1].legend()

    # %% NUMERICAL SUMMARY
    print('8.2 - Integrate a rectangular aperture using explicit two-dimensional Simpson weights')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
