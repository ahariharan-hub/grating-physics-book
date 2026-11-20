# -*- coding: utf-8 -*-
"""5.3, worked example 1: Construct a reciprocal basis for a nonorthogonal real-space lattice.

Companion topic: Two translation symmetries and two integers.
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
direct = np.array([[4.0, 1.0], [0.0, 5.0]])  # columns are lattice vectors
show_plot = True

# %% SETUP: coordinates, sampling and constants
assert abs(np.linalg.det(direct)) > 0

# %% NUMERICAL CALCULATION
reciprocal = 2 * np.pi * np.linalg.inv(direct).T
orthogonality = direct.T @ reciprocal
labels = np.array([(m, n) for m in range(-2, 3) for n in range(-2, 3)])
points = labels @ reciprocal.T

# %% CHECK AND INTERPRET
assert np.allclose(orthogonality, 2 * np.pi * np.eye(2))
print("a_i dot b_j matrix:", orthogonality)

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.scatter(points[:, 0], points[:, 1])
    ax.set(xlabel='Reciprocal kx (rad/um)', ylabel='Reciprocal ky (rad/um)', aspect='equal')

    # %% NUMERICAL SUMMARY
    print('5.3 - Construct a reciprocal basis for a nonorthogonal real-space lattice')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
