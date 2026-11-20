# -*- coding: utf-8 -*-
"""2.4, worked example 2: Intersect outgoing rays with a sphere and a detector plane.

Companion topic: Where does one order meet each surface?.
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
period = 2.5
radius = 2.0
plane_distance = 3.0
show_plot = True

# %% SETUP: coordinates, sampling and constants
orders = np.arange(-2, 3)
alpha = orders * wavelength / period
gamma = np.sqrt(1 - alpha**2)

# %% NUMERICAL CALCULATION
sphere_x = radius * alpha
sphere_s = radius * gamma
plane_x = plane_distance * alpha / gamma

# %% CHECK AND INTERPRET
assert np.allclose(sphere_x**2 + sphere_s**2, radius**2)
print("Sphere projections are proportional to sine; plane intersections to tangent.")

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for sx, ss, px in zip(sphere_x, sphere_s, plane_x):
        ax.plot([0, px], [0, plane_distance], color='gray')
        ax.plot(sx, ss, 'o', color='blue')
        ax.plot(px, plane_distance, 's', color='orange')
    ax.set(xlabel='x (um)', ylabel='Downstream distance s (um)', aspect='equal')

    # %% NUMERICAL SUMMARY
    print('2.4 - Intersect outgoing rays with a sphere and a detector plane')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
