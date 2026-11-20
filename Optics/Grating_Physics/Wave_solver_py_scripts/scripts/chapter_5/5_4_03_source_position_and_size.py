# -*- coding: utf-8 -*-
"""5.4, worked example 3: Distinguish moving a source point from widening a source disk.

Companion topic: Distinguish a source position from source size.
Air exterior; angular source and collection geometry only. Distinct source points would add incoherently in an imaging calculation.
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
vacuum_wavelength = 0.193
period = 0.6
NA = 0.75
source_radius = 0.45
source_center_x = 0.2
show_plot = True

# %% SETUP: coordinates, sampling and constants
grid = np.linspace(-source_radius, source_radius, 31)
xx, yy = np.meshgrid(grid, grid)
inside = xx**2 + yy**2 <= source_radius**2
source_points = np.column_stack([xx[inside] + source_center_x, yy[inside]])
orders = np.arange(-4, 5)

# %% NUMERICAL CALCULATION
order_x = source_points[:, 0, None] + orders * vacuum_wavelength / (period * NA)
order_y = np.broadcast_to(source_points[:, 1, None], order_x.shape)
collected = order_x**2 + order_y**2 <= 1
geometric_collection_fraction = collected.mean(axis=0)

# %% CHECK AND INTERPRET
assert np.all((geometric_collection_fraction >= 0) & (geometric_collection_fraction <= 1))
print("Fractions count equally weighted source samples; they are not diffraction efficiencies.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].scatter(source_points[:, 0], source_points[:, 1], s=6)
    axes[0].set(xlabel='Source sigma x', ylabel='Source sigma y', aspect='equal')
    axes[1].bar(orders, geometric_collection_fraction)
    axes[1].set(xlabel='Order m', ylabel='Fraction of sampled source points collected')

    # %% NUMERICAL SUMMARY
    print('5.4 - Distinguish moving a source point from widening a source disk')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
