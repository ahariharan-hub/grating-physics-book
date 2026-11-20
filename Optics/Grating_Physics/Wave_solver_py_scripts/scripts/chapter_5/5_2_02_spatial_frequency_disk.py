# -*- coding: utf-8 -*-
"""5.2, worked example 2: Calculate the propagating spatial-frequency disk in different media.

Companion topic: Index and the spatial-frequency disk.
Vacuum wavelength and lengths um. Homogeneous lossless exterior media; only coordinate geometry is calculated.
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
vacuum_wavelength = 0.6
indices = [1.0, 1.5, 2.0]
show_plot = True

# %% SETUP: coordinates, sampling and constants
azimuth = np.linspace(0, 2 * np.pi, 361)

# %% NUMERICAL CALCULATION
radii = np.array(indices) / vacuum_wavelength
circles = [radius * np.column_stack([np.cos(azimuth), np.sin(azimuth)]) for radius in radii]

# %% CHECK AND INTERPRET
assert np.allclose(radii * (vacuum_wavelength / np.array(indices)), 1)
print("Frequency disk radius is n/lambda0, not an index-independent constant.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    for index, circle in zip(indices, circles):
        ax.plot(circle[:, 0], circle[:, 1], label=f'n={index:g}')
    ax.set(xlabel='fx (cycles/um)', ylabel='fy (cycles/um)', aspect='equal')
    ax.legend()

    # %% NUMERICAL SUMMARY
    print('5.2 - Calculate the propagating spatial-frequency disk in different media')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
