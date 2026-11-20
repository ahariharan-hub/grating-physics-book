# -*- coding: utf-8 -*-
"""4.1, worked example 3: Verify that a decaying oscillatory field still satisfies Helmholtz.

Companion topic: Constructing an evanescent solution.
Medium wavelength and lengths um. Downstream s=-z; U=exp(i kx x+i q s); positive-imaginary q decays.
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
transverse_ratio = 1.4
grid_step = 0.005
show_plot = True

# %% SETUP: coordinates, sampling and constants
k = 2 * np.pi / wavelength
kx = k * transverse_ratio
kappa = np.sqrt(kx**2 - k**2)
x = np.arange(-1, 1 + grid_step / 2, grid_step)
s = np.arange(0, 1 + grid_step / 2, grid_step)

# %% NUMERICAL CALCULATION
field = np.exp(1j * kx * x[None, :] - kappa * s[:, None])
second_x = (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / grid_step**2
second_s = (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / grid_step**2
residual = second_x + second_s + k**2 * field[1:-1, 1:-1]
normalized = abs(residual) / (k**2 * abs(field[1:-1, 1:-1]))

# %% CHECK AND INTERPRET
assert np.max(normalized) < 0.001
print("Maximum normalized finite-difference residual:", np.max(normalized))

# %% GRAPHICS
if __name__ == '__main__':
    plt = pyplot(show_plot)
    figure, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(s[1:-1], normalized.mean(axis=1))
    ax.set(xlabel='Depth (um)', ylabel='Mean normalized Helmholtz residual')

    # %% NUMERICAL SUMMARY
    print('4.1 - Verify that a decaying oscillatory field still satisfies Helmholtz')
    print('Numerical check passed. Read CHECK AND INTERPRET before changing parameters.')
    finish(figure, show_plot)
