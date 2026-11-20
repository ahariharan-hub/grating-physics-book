# -*- coding: utf-8 -*-
"""8.2, worked example 2: Refine the aperture quadrature independently of observation sampling.

Companion topic: Numerical convergence of the RS I patch sum.
Full 3D RS I kernel integrated over a prescribed square aperture. Medium wavelength and lengths um.
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
aperture_width = 6.0
distance = 1.5
point_counts = [9, 17, 33, 65, 129]
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
fields = []

# %% NUMERICAL CALCULATION
for count in point_counts:
    coordinate = np.linspace(-aperture_width / 2, aperture_width / 2, count)
    xx, yy = np.meshgrid(coordinate, coordinate)
    step = coordinate[1] - coordinate[0]
    weights = np.ones(count)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    radius = np.sqrt(xx**2 + yy**2 + distance**2)
    kernel = distance * np.exp(1j * k * radius) * (1 - 1j * k * radius) / (2 * np.pi * radius**3)
    fields.append(np.sum(kernel * np.outer(weights, weights)) * step**2 / 9)
fields = np.array(fields)
relative_differences = abs(fields - fields[-1]) / abs(fields[-1])

# %% CHECK AND INTERPRET
assert relative_differences[-2] < relative_differences[0]
print("Relative differences:", relative_differences)
print("The densest quadrature is a numerical reference, not exact truth.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.loglog(point_counts[:-1], np.maximum(relative_differences[:-1], 1e-16), 'o-')
    ax.set(xlabel='Aperture points per axis', ylabel='Relative on-axis complex-field difference')

    # %% NUMERICAL SUMMARY
    print('8.2 - Refine the aperture quadrature independently of observation sampling')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
